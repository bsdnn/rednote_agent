from __future__ import annotations
import json
import re
import asyncio
import logging
from typing import AsyncIterator

from .config import settings
from .prompts import (
    SYSTEM_PROMPT,
    PLANNING_PROMPT,
    REFLECTION_PROMPT,
    build_user_message,
    build_planning_message,
    build_reflection_message,
    build_refine_message,
)
from ..models.request import GenerateRequest, RefineRequest
from ..models.response import GenerateResponse
from ..services.deepseek_client import get_client
from ..services.tools_registry import AVAILABLE_TOOLS, TOOLS_DEFINITION
from ..services.memory_service import save_copy_result

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine to run in the background without blocking.

    Keeps a reference to prevent GC and logs any exception that escapes.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.error("Background task failed: %s", t.exception())

    task.add_done_callback(_done)


def _parse_json_response(content: str) -> dict:
    """Strip markdown fences and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?\n?", "", content).strip("` \n")
    return json.loads(content)


async def _planning_phase(
    client,
    query: str,
    tone: str,
    persona,
    user_id: str | None,
) -> dict:
    response = await client.chat(
        messages=[
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user", "content": build_planning_message(query, tone, persona, user_id)},
        ],
        tools=None,
    )
    content = response["choices"][0]["message"]["content"]
    return _parse_json_response(content)


async def _reflection_phase(client, draft: GenerateResponse) -> dict:
    response = await client.chat(
        messages=[
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": build_reflection_message(draft)},
        ],
        tools=None,
    )
    content = response["choices"][0]["message"]["content"]
    return _parse_json_response(content)


async def generate_rednote(
    request: GenerateRequest, *, enable_planning: bool = True
) -> AsyncIterator[dict]:
    client = get_client()

    # Phase 1: Planning (skippable for A/B experiment)
    plan: dict | None = None
    if enable_planning:
        yield {"event": "agent_thinking", "data": {"step": "制定任务计划中...", "iteration": 0}}
        try:
            plan = await _planning_phase(
                client, request.query, str(request.tone.value), request.persona, request.user_id
            )
            yield {"event": "agent_plan", "data": plan}
            logger.info("Plan created: %s", plan.get("goal"))
        except Exception as e:
            logger.warning("Planning phase failed: %s — continuing without plan", e)

    # Build initial user message, injecting plan + user_id hint
    user_content = build_user_message(request.query, request.tone, request.persona)
    if plan:
        user_content += f"\n\n[执行计划]\n{json.dumps(plan, ensure_ascii=False)}"
    if request.user_id:
        user_content += (
            f"\n\n当前用户ID为「{request.user_id}」，"
            "请优先调用 recall_user_history 工具查询其历史偏好，再开始创作。"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    logger.info("Starting generation: query='%s' tone='%s'", request.query, request.tone)

    persona_json: str | None = None
    if request.user_id and request.persona:
        persona_json = request.persona.model_dump_json()

    async for event in _agent_loop(
        client,
        messages,
        request.max_iterations,
        user_id=request.user_id,
        query=request.query,
        persona_json=persona_json,
    ):
        yield event


async def refine_rednote(request: RefineRequest) -> AsyncIterator[dict]:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *request.conversation_history,
        {
            "role": "assistant",
            "content": json.dumps(request.previous_result, ensure_ascii=False),
        },
        {"role": "user", "content": build_refine_message(request.refinement_instruction)},
    ]
    logger.info("Starting refinement: instruction='%s'", request.refinement_instruction)
    async for event in _agent_loop(client, messages, max_iterations=3):
        yield event


async def _agent_loop(
    client,
    messages: list[dict],
    max_iterations: int,
    *,
    user_id: str | None = None,
    query: str = "",
    persona_json: str | None = None,
) -> AsyncIterator[dict]:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    _json_retried = False
    _reflection_count = 0
    best_draft: GenerateResponse | None = None
    best_score: int = 0
    _tool_cache: dict[tuple[str, str], str] = {}

    for i in range(max_iterations):
        yield {
            "event": "agent_thinking",
            "data": {"step": f"AI正在思考中（第{i + 1}轮）...", "iteration": i + 1},
        }

        try:
            response = await client.chat(messages, tools=TOOLS_DEFINITION)
        except Exception as e:
            logger.error("API request failed: %s", e)
            yield {"event": "error", "data": {"message": f"API请求失败: {e}", "code": 503}}
            return

        if "choices" not in response:
            yield {
                "event": "error",
                "data": {"message": f"无效的API响应: {response}", "code": 500},
            }
            return

        usage = response.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        response_message = response["choices"][0]["message"]

        # ── Tool calls ────────────────────────────────────────────────────────
        if response_message.get("tool_calls"):
            messages.append(response_message)
            _json_retried = False

            tool_calls = response_message["tool_calls"]
            tool_names_preview = "、".join(tc["function"]["name"] for tc in tool_calls)
            yield {
                "event": "agent_thinking",
                "data": {"step": f"并行调用工具：{tool_names_preview}"},
            }

            async def _exec_tool(tool_call: dict) -> tuple[dict, str, str, bool]:
                func_name = tool_call["function"]["name"]
                try:
                    func_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                args_key = json.dumps(func_args, sort_keys=True, ensure_ascii=False)
                cache_key = (func_name, args_key)
                if cache_key in _tool_cache:
                    return tool_call, func_name, _tool_cache[cache_key], True

                if func_name not in AVAILABLE_TOOLS:
                    return tool_call, func_name, f"工具 '{func_name}' 不存在", False

                try:
                    result = await asyncio.wait_for(
                        AVAILABLE_TOOLS[func_name](**func_args),
                        timeout=settings.TOOL_TIMEOUT_SECONDS,
                    )
                    result_str = str(result)
                    _tool_cache[cache_key] = result_str
                    return tool_call, func_name, result_str, False
                except asyncio.TimeoutError:
                    logger.warning("Tool '%s' timed out", func_name)
                    return (
                        tool_call,
                        func_name,
                        f"工具 '{func_name}' 超时（{settings.TOOL_TIMEOUT_SECONDS:.0f}秒），跳过。",
                        False,
                    )

            results = await asyncio.gather(*[_exec_tool(tc) for tc in tool_calls])

            for tool_call, func_name, result_str, cached in results:
                logger.info(
                    "Tool '%s' result: %d chars (cached=%s)",
                    func_name,
                    len(result_str),
                    cached,
                )
                yield {
                    "event": "tool_result",
                    "data": {
                        "tool": func_name,
                        "summary": result_str[:200],
                        "cached": cached,
                    },
                }
                messages.append(
                    {
                        "role": "tool",
                        "content": result_str,
                        "tool_call_id": tool_call["id"],
                    }
                )
            continue

        # ── Text response (draft output) ──────────────────────────────────────
        if response_message.get("content"):
            content = response_message["content"]
            try:
                raw = json.loads(content)
                result = GenerateResponse.model_validate(raw)

                # Phase 3: Reflection (up to MAX_REFLECTIONS cycles)
                if _reflection_count < settings.MAX_REFLECTIONS:
                    _reflection_count += 1
                    yield {
                        "event": "agent_thinking",
                        "data": {
                            "step": f"自我审核中（第{_reflection_count}/{settings.MAX_REFLECTIONS}次）...",
                            "iteration": i + 1,
                        },
                    }
                    try:
                        critique = await _reflection_phase(client, result)
                        min_score = critique.get("min_score", 10)
                        logger.info(
                            "Reflection scores — virality=%s tone=%s accuracy=%s min=%s",
                            critique.get("virality_score"),
                            critique.get("tone_match_score"),
                            critique.get("accuracy_score"),
                            min_score,
                        )

                        # Track best-scored draft so we never regress on rewrite
                        if best_draft is None or min_score > best_score:
                            best_draft = result
                            best_score = min_score

                        if min_score < settings.REFLECTION_MIN_SCORE:
                            issues = "；".join(critique.get("issues", []))
                            suggestions = "；".join(critique.get("suggestions", []))
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"审核发现以下问题：{issues}。"
                                    f"改进建议：{suggestions}。"
                                    "请修改后重新输出完整JSON。"
                                ),
                            })
                            yield {
                                "event": "agent_thinking",
                                "data": {"step": "发现改进点，正在修改...", "iteration": i + 1},
                            }
                            continue
                    except Exception as e:
                        logger.warning("Reflection phase failed: %s — accepting draft as-is", e)

                # Prefer the highest-scored draft we've certified, fall back to current
                final_result = best_draft if best_draft is not None else result

                # Save to memory in background (fire-and-forget)
                if user_id:
                    _fire_and_forget(
                        save_copy_result(user_id, query, final_result.title, persona_json)
                    )

                total_tokens = total_prompt_tokens + total_completion_tokens
                logger.info(
                    "Generation complete — prompt=%d completion=%d total=%d tokens",
                    total_prompt_tokens,
                    total_completion_tokens,
                    total_tokens,
                )
                yield {"event": "complete", "data": final_result.model_dump()}
                yield {
                    "event": "token_usage",
                    "data": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
                return

            except json.JSONDecodeError:
                if not _json_retried:
                    _json_retried = True
                    logger.warning("JSON parse failed, retrying with correction prompt")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "你的输出不是合法JSON，请只输出JSON对象，不要其他文字。",
                    })
                    yield {
                        "event": "agent_thinking",
                        "data": {"step": "JSON格式有误，正在修正..."},
                    }
                    continue
                else:
                    yield {
                        "event": "error",
                        "data": {"message": "响应JSON格式无效，无法解析", "code": 422},
                    }
                    return
            except Exception as e:
                logger.error("Response validation failed: %s", e)
                yield {
                    "event": "error",
                    "data": {"message": f"响应解析失败: {e}", "code": 422},
                }
                return

    # Reached max_iterations without a clean return — fall back to best draft if any
    if best_draft is not None:
        logger.warning(
            "Max iterations reached — returning best-scored draft (score=%d)", best_score
        )
        if user_id:
            _fire_and_forget(
                save_copy_result(user_id, query, best_draft.title, persona_json)
            )
        yield {"event": "complete", "data": best_draft.model_dump()}
        yield {
            "event": "warning",
            "data": {
                "message": f"已达最大迭代次数，返回当前最高分草稿（审核分：{best_score}）",
            },
        }
        total_tokens = total_prompt_tokens + total_completion_tokens
        yield {
            "event": "token_usage",
            "data": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return

    yield {
        "event": "error",
        "data": {"message": "已达到最大迭代次数，生成失败", "code": 500},
    }
