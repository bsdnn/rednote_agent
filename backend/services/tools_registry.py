from .rag_service import query_product_database
from .search_service import search_web, get_trending_topics, fetch_webpage
from .memory_service import recall_user_history

AVAILABLE_TOOLS: dict = {
    "query_product_database": query_product_database,
    "search_web": search_web,
    "get_trending_topics": get_trending_topics,
    "fetch_webpage": fetch_webpage,
    "recall_user_history": recall_user_history,
}

TOOLS_DEFINITION: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "recall_user_history",
            "description": (
                "Look up a user's past persona and recent copy history from persistent memory. "
                "Call this first when a user_id is available to personalize the output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique identifier",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_product_database",
            "description": (
                "Semantic search over the local skincare product database. "
                "Provide a user need or pain point (e.g. '去痘印', '补水保湿') "
                "and the top 3 best-matching products are returned with details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User skin concern or product need in Chinese or English",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_topics",
            "description": (
                "Fetch current trending keywords on Xiaohongshu for a given product category. "
                "Use this to make copy feel timely and relevant. Call at most once per generation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category, e.g. '护肤', '美白', '防晒', '抗老'",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for the latest product reviews, trends, or ingredient info. "
                "Returns titles and snippets. Use fetch_webpage on a result URL for full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch and extract the full text content of a webpage URL. "
                "Use after search_web when you need deeper detail from a specific page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]
