# RAG v3 Eval Report — 2026-05-30

## Overall Metrics

| Config | N | Recall@3 | Recall@10 | MRR | Faithfulness | Avg sec | Forbidden |
|---|---|---|---|---|---|---|---|
| C0 | 40 | 0.665 | 0.665 | 0.475 | 7.075 | 0.10 | 0 |
| C1 | 40 | 0.595 | 0.595 | 0.362 | 6.375 | 0.08 | 0 |
| C2 | 40 | 0.660 | 0.660 | 0.467 | 6.9 | 9.69 | 0 |
| C3 | 40 | 0.660 | 0.660 | 0.467 | 7.1 | 10.20 | 0 |
| C4 | 40 | 0.648 | 0.648 | 0.442 | 6.925 | 10.63 | 0 |

## Per-Category Breakdown (Recall@3)

| Category | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| adversarial| 1.000| 1.000| 1.000| 1.000| 1.000 |
| colloquial| 0.200| 0.200| 0.200| 0.200| 0.200 |
| cross_category| 0.667| 0.667| 0.667| 0.667| 0.667 |
| direct_need| 0.333| 0.333| 0.417| 0.417| 0.417 |
| ingredient_lookup| 0.757| 0.524| 0.701| 0.701| 0.701 |
| persona_strong| 0.917| 0.917| 0.917| 0.917| 0.833 |
| synonym| 0.750| 0.750| 0.750| 0.750| 0.750 |

## Delta vs Baseline (C0)

| Config | ΔRecall@3 | ΔMRR | ΔFaithfulness |
|---|---|---|---|
| C1 | -0.070 | -0.112 | -0.70 |
| C2 | -0.004 | -0.008 | -0.17 |
| C3 | -0.004 | -0.008 | +0.02 |
| C4 | -0.017 | -0.033 | -0.15 |

## Dual-Judge Consistency (Custom-1to10 vs LI-RelevancyEvaluator)

Custom score >= 7 treated as PASS. Cohen's κ measures agreement beyond chance.

| Config | Both PASS | Both FAIL | Custom PASS / LI FAIL | Custom FAIL / LI PASS | Cohen κ |
|---|---|---|---|---|---|
| C0 | 28 | 6 | 0 | 6 | +0.583 |
| C1 | 24 | 7 | 2 | 7 | +0.461 |
| C2 | 27 | 7 | 1 | 5 | +0.605 |
| C3 | 27 | 8 | 1 | 4 | +0.679 |
| C4 | 26 | 8 | 1 | 5 | +0.628 |

## Failure Analysis (faithfulness ≤ 5)

### C0 — 10 low-score records

**adversarial** (4 records)
- `Q011` _美白牙膏_ (faith=1)
  > Query是关于美白牙膏，但context全部是面部护肤产品（精华、磨砂膏）和美白方法，未提及任何牙膏或口腔护理相关内容，完全无关。
- `Q012` _汽车防晒膜_ (faith=1)
  > Query是关于汽车防晒膜，而Context内容均为个人护肤防晒和晒后修复，与汽车防晒膜完全无关。

**colloquial** (2 records)
- `Q020` _黑眼圈怎么淡化_ (faith=5)
  > 匹配1直接相关，针对黑眼圈；匹配2和3讨论美白淡斑，与黑眼圈间接相关，但非直接针对。整体部分相关，信息不充分。
- `Q021` _唇部干裂起皮用啥_ (faith=1)
  > Context 讨论的是脸部皮肤干燥起皮，而 query 特指唇部干裂起皮，两者部位不同，context 未涉及唇部护理，因此完全不相关。

**cross_category** (2 records)
- `Q022` _卸妆水和卸妆油区别_ (faith=3)
  > context 仅匹配1提到卸妆产品（卸妆水/油），但未比较两者区别；匹配2和匹配3讨论的是屏障修复，与卸妆水/油区别完全无关。整体信息不足且部分不相关。
- `Q023` _洁面和洗面奶哪个温和_ (faith=1)
  > Context discusses skincare ingredients like sodium lauroyl glutamate, bearberry extract, and arbutin, but does not compare '洁面' (cleansing products) and '洗面奶' (facial cleanser) in terms of gentleness. The query asks about the difference between two types of cleansing products, while context is about unrelated topics.

**ingredient_lookup** (1 records)
- `Q034` _含咖啡因的眼霜_ (faith=5)
  > 匹配1与查询相关，包含咖啡因和眼霜，但未明确提及‘含咖啡因的眼霜’这一产品；匹配2和匹配3与查询无关。整体相关性不足。

**persona_strong** (1 records)
- `Q040` _便宜大碗的身体乳_ (faith=5)
  > 匹配1提到身体乳且含便宜成分，但未强调价格；匹配2和3主要讨论面部护肤，与身体乳无关。整体部分相关但信息不足。

### C1 — 12 low-score records

**adversarial** (4 records)
- `Q011` _美白牙膏_ (faith=1)
  > Context 讨论的是面部护肤（早C晚A、祛痘印、身体磨砂），与查询的‘美白牙膏’完全无关。
- `Q012` _汽车防晒膜_ (faith=1)
  > Context 讨论的是防晒霜、护肤成分和晒后修复，与汽车防晒膜完全无关。

**colloquial** (1 records)
- `Q021` _唇部干裂起皮用啥_ (faith=1)
  > Query是关于唇部干裂起皮的产品推荐，但context全部是关于脸部皮肤干燥起皮的处理方法，没有提到唇部护理。因此完全不相关。

**cross_category** (2 records)
- `Q022` _卸妆水和卸妆油区别_ (faith=1)
  > context 内容为卸妆产品和护肤建议，未提及卸妆水和卸妆油的区别。
- `Q023` _洁面和洗面奶哪个温和_ (faith=5)
  > 匹配2提到了温和清洁，与query‘温和’相关，但未直接比较洁面和洗面奶；匹配1和3不相关。总体部分相关。

**direct_need** (3 records)
- `Q014` _混油皮夏天用的清爽防晒_ (faith=5)
  > Context 提到了防晒，但主要讨论屏障修复和控油，未具体针对混油皮和清爽型防晒，相关性部分满足。
- `Q038` _身体磨砂膏推荐_ (faith=5)
  > 只有匹配2直接提及身体磨砂膏（咖啡颗粒去角质），但不够具体；匹配1和匹配3与身体磨砂膏无关。整体信息部分相关但不足。

**persona_strong** (1 records)
- `Q040` _便宜大碗的身体乳_ (faith=1)
  > 所有检索结果中，只有[prod_18]提及身体乳，但描述的是非洲牛油果脂与椰子油深层滋润，并未提及'便宜大碗'这一核心需求；其他结果完全不相关。因此context与query相关性极低。

**synonym** (1 records)
- `Q029` _护肤品_ (faith=5)
  > Context 提到护肤品成分和使用方法，与query '护肤品'相关，但内容聚焦于具体产品使用建议，未全面覆盖护肤品的一般性知识或定义，信息不够充分。

### C2 — 12 low-score records

**adversarial** (4 records)
- `Q011` _美白牙膏_ (faith=1)
  > Context 讨论的是烟酰胺、维C、377等美白成分的使用注意事项和推荐，但未提及任何关于美白牙膏的信息，与 query '美白牙膏' 完全无关。
- `Q012` _汽车防晒膜_ (faith=1)
  > Context 主要讨论A醇、防晒霜和抗老护肤，与汽车防晒膜完全无关。

**colloquial** (2 records)
- `Q020` _黑眼圈怎么淡化_ (faith=4)
  > Query asks about dark circles; context mostly discusses whitening, spots, and general skincare, with only one match partially addressing dark circles (post_93 mentions eye area but not directly). Insufficient specific info on dark circles.
- `Q021` _唇部干裂起皮用啥_ (faith=3)
  > 用户查询针对唇部干裂起皮，但context均讨论面部皮肤干燥起皮，未涉及唇部护理。内容虽相关于干燥起皮，但部位不匹配，相关性较低。

**cross_category** (2 records)
- `Q022` _卸妆水和卸妆油区别_ (faith=1)
  > Context 中没有任何关于卸妆水和卸妆油区别的内容，只提到卸妆产品（如米糠油卸妆），且其他匹配内容涉及抗老和敏感肌护理，与 query 完全无关。
- `Q023` _洁面和洗面奶哪个温和_ (faith=1)
  > Context 1 是关于洁面产品的温和性，但 query 是询问洁面和洗面奶哪个更温和，context 未直接比较两者；Context 2 和 3 讨论烟酰胺和维C，与 query 完全无关。整体相关性低。

**direct_need** (2 records)
- `Q038` _身体磨砂膏推荐_ (faith=1)
  > Context 提及的是咖啡颗粒磨砂膏，但未提及身体磨砂膏推荐；其他匹配为积雪草成分介绍和A醇使用注意事项，均与身体磨砂膏推荐无关。
- `Q039` _发质毛躁修护_ (faith=5)
  > Context 包含一个针对毛躁发质的修护产品（匹配1），与query部分相关，但其他匹配内容完全无关（匹配2和3），且信息不够充分，无法完全回答query。

**ingredient_lookup** (1 records)
- `Q036` _氨基酸洁面适合什么肤质_ (faith=5)
  > 仅匹配1直接相关，回答氨基酸洁面适合干皮、敏感肌及孕妇，但未覆盖其他肤质；匹配2和3与query无关。

**persona_strong** (1 records)
- `Q040` _便宜大碗的身体乳_ (faith=3)
  > Query 要求便宜大碗的身体乳，但context中只有[匹配1]是身体乳产品，且未提及价格和容量；[匹配2]和[匹配3]分别讨论A醇使用和面部精华，与身体乳无关。整体相关性低。

### C3 — 10 low-score records

**adversarial** (4 records)
- `Q011` _美白牙膏_ (faith=1)
  > Context discusses skincare ingredients like niacinamide, vitamin C, and 377 for brightening, but does not mention toothpaste or teeth whitening. Query is about whitening toothpaste, which is unrelated.
- `Q012` _汽车防晒膜_ (faith=1)
  > Context 内容主要讨论A醇、防晒霜和抗老护肤，与汽车防晒膜完全无关。

**colloquial** (1 records)
- `Q021` _唇部干裂起皮用啥_ (faith=2)
  > Query 是关于唇部干裂起皮，而 context 全部是关于面部皮肤干燥起皮，未提及唇部护理，因此相关性很低。

**cross_category** (2 records)
- `Q022` _卸妆水和卸妆油区别_ (faith=2)
  > Context 仅提及卸妆（如匹配1中的卸妆产品），但未解释卸妆水和卸妆油的区别，主要讨论抗老、敏感肌护理等无关话题。
- `Q023` _洁面和洗面奶哪个温和_ (faith=5)
  > Context 中只有匹配1直接相关，提供了洁面产品的温和性信息，但未明确对比洁面和洗面奶的温和性。匹配2和3讨论烟酰胺和维C，与query无关。整体部分相关且信息不足。

**direct_need** (1 records)
- `Q039` _发质毛躁修护_ (faith=5)
  > 匹配1与query相关，提供了修护毛躁发质的成分和方法，但匹配2和匹配3完全不相关，因此整体相关性中等。

**ingredient_lookup** (1 records)
- `Q034` _含咖啡因的眼霜_ (faith=5)
  > 匹配1提到咖啡因和眼霜，但主要针对细纹、黑眼圈和眼袋，未明确提及咖啡因含量或功效；匹配2提及咖啡因眼霜，但主体是茶多酚；匹配3是身体磨砂膏，与眼霜无关。整体部分相关，信息不足。

**persona_strong** (1 records)
- `Q040` _便宜大碗的身体乳_ (faith=1)
  > Context 描述的是身体乳和护肤品，但 query 要求的是“便宜大碗的身体乳”，而 context 中只有第一个匹配是身体乳，但未提及价格和容量，且其他匹配完全不相关。整体未能回答 query 的核心需求。

### C4 — 12 low-score records

**adversarial** (4 records)
- `Q011` _美白牙膏_ (faith=1)
  > Context discusses skincare ingredients like niacinamide, vitamin C, and 377 for brightening, but does not mention whitening toothpaste at all.
- `Q012` _汽车防晒膜_ (faith=1)
  > 所有context均未提及汽车防晒膜，而是讨论护肤品（A醇、防晒霜、早C晚A），与query完全无关。

**colloquial** (2 records)
- `Q020` _黑眼圈怎么淡化_ (faith=3)
  > Context mentions black circles (黑眼圈) only in the third match, but the main focus is on whitening and skincare routines, not specifically on how to lighten dark circles. The information is partial and not directly answering the query.
- `Q021` _唇部干裂起皮用啥_ (faith=3)
  > Contexts focus on facial dryness and makeup issues, but query specifically asks about chapped and peeling lips. No lip-specific product or advice is provided.

**cross_category** (2 records)
- `Q022` _卸妆水和卸妆油区别_ (faith=3)
  > 只有匹配1提到卸妆产品，但未对比卸妆水和卸妆油的区别；匹配2和匹配3完全不相关。
- `Q023` _洁面和洗面奶哪个温和_ (faith=1)
  > Query询问洁面和洗面奶哪个温和，但context内容涉及产品成分和护肤建议，未直接比较洁面与洗面奶的温和性，因此完全不相关。

**direct_need** (2 records)
- `Q038` _身体磨砂膏推荐_ (faith=5)
  > 只有[匹配1]明确提到了身体磨砂膏产品，包含咖啡颗粒去角质等符合query的信息；[匹配2]和[匹配3]分别是积雪草成分介绍和A醇使用贴士，与身体磨砂膏无直接关系。整体相关但信息不充分。
- `Q039` _发质毛躁修护_ (faith=1)
  > Context主要讨论A醇使用注意事项和护肤步骤，与用户query'发质毛躁修护'完全无关。

**ingredient_lookup** (1 records)
- `Q036` _氨基酸洁面适合什么肤质_ (faith=3)
  > 匹配1部分相关（提及干皮、敏感肌），但未直接回答氨基酸洁面适合什么肤质；匹配2和3完全不相关（分别关于透明质酸和乳酸）。整体相关性较低。

**persona_strong** (1 records)
- `Q040` _便宜大碗的身体乳_ (faith=1)
  > Context discusses A醇、烟酰胺等面部护肤，与query“便宜大碗的身体乳”完全无关。

