# hiver-sde-agent

# Hiver — AmazonHelp Support Agent

## 1. Problem Framing

This project builds an AI support agent for **AmazonHelp** using the Customer Support on Twitter (TWCS) dataset.

The agent is designed to:

1. Classify an incoming customer message into a small, data-derived intent taxonomy.
2. Draft a reply grounded in historically similar AmazonHelp resolutions.
3. Decide whether the message should be **AUTO-HANDLE** or **HUMAN**, with an explicit reason.

For this project, “good” means:

* intents are understandable and supported by the AmazonHelp data;
* similar historical cases can be retrieved as evidence;
* intent predictions outperform simple baselines on a held-out evaluation split;
* escalation decisions are explicit and auditable;
* responses avoid inventing unsupported policies, refunds, delivery dates, or guarantees.

The implementation prioritizes the core support-agent loop rather than production infrastructure or a complex UI.

---

## 2. What I Built

```text
TWCS data
   |
   v
AmazonHelp brand selection
   |
   v
Conversation reconstruction
   |
   v
Intent discovery + frozen 10-intent taxonomy
   |
   v
Human-verified golden set (191 examples)
   |
   +----------------------------+
   |                            |
   v                            v
Intent classifier          Historical retrieval
   |                            |
   +-------------+--------------+
                 |
                 v
          Escalation decision
                 |
                 v
          AUTO-HANDLE / HUMAN
                 |
                 v
          Grounded draft response
```

Core tools:

* Python
* pandas / NumPy
* scikit-learn
* sentence-transformers (`all-MiniLM-L6-v2`)
* FAISS
* Transformers
* Gemini API for response generation when quota is available

---

## 3. Dataset and Brand Selection

The primary dataset is Customer Support on Twitter (TWCS), containing noisy, multi-turn customer-support conversations across many brands.

I selected **AmazonHelp** because it provided a large directly linked support corpus among the leading candidates examined, including:

* 169,840 outbound AmazonHelp responses;
* 168,814 directly linked customer-response interactions;
* 71,048 distinct customers;
* high customer-message diversity;
* a large historical response corpus suitable for retrieval.

The selection was based on corpus size, customer coverage, message diversity, and availability of historical resolutions.

---

## 4. Conversation Reconstruction

The raw TWCS records were converted into customer → AmazonHelp response pairs using the dataset's tweet/thread relationship fields.

The resulting AmazonHelp conversation dataset contains approximately 168k customer-response pairs with both customer and agent text available.

Validation of the reconstructed artifact showed:

* 168,814 customer-response pairs;
* 71,048 unique customers;
* 153,004 unique customer messages;
* 0 missing customer messages;
* 0 missing agent responses;
* 0 duplicate pairs.

This reconstructed dataset is used for intent discovery and historical-resolution retrieval.

---

## 5. Intent Taxonomy

I froze the following 10 intents after inspecting sampled AmazonHelp conversations and cluster examples:

| Intent                             | Definition                                                        |
| ---------------------------------- | ----------------------------------------------------------------- |
| `Order_Delivery_Tracking`          | Customer wants order/package location or status.                  |
| `Delivery_Delay_or_Missed_Date`    | Order is late, delayed, or misses an expected/promised date.      |
| `Order_Cancellation_or_Change`     | Customer wants to cancel/change an order or reports cancellation. |
| `Missing_Wrong_or_Damaged_Item`    | Missing, wrong, incomplete, or damaged item/package.              |
| `Returns_Refunds_or_Replacements`  | Return, refund, or replacement request.                           |
| `Payment_or_Charge_Issue`          | Payment, charge, billing, or cashback issue.                      |
| `Account_Login_or_Security`        | Account access, verification, login, or security issue.           |
| `Product_Availability_or_Pricing`  | Product price, availability, stock, promotion, or discount.       |
| `Product_or_Service_Information`   | General product/service information, compatibility, or usage.     |
| `Prime_Video_Content_Availability` | Prime Video movie/show/season/episode/content availability.       |

The most important confusable boundaries are:

* delivery tracking vs delivery delay;
* missing/wrong/damaged vs returns/replacements;
* product/service information vs more specialized intents.

---

## 6. Golden Evaluation Set

I created a 200-example candidate pool and completed **191 human-verified examples**. Nine incomplete/ambiguous records were excluded rather than assigning unsupported labels.

Checks:

* 191 final examples;
* all 10 intents represented;
* 0 missing intent labels;
* 0 missing escalation labels;
* 0 duplicate customer messages;
* escalation labels: 129 `False`, 62 `True`.

### Sampling and Labeling Note

Candidates were sampled from the AmazonHelp customer-message corpus after conversation reconstruction and intent discovery.

AI/rule suggestions were used only as **pre-labels** to reduce manual effort. The final `true_intent` and `true_escalation` fields were human-verified. Provisional AI suggestions were not treated as ground truth.

---

## 7. Intent Baselines and Results

All intent experiments use the same stratified 75/25 split with `random_state=42`. The holdout contains 48 examples.

| Model                                  |  Accuracy |  Macro F1 | Weighted F1 |
| -------------------------------------- | --------: | --------: | ----------: |
| Majority baseline                      |     35.4% |     0.052 |           — |
| TF-IDF + Logistic Regression           |     27.1% |     0.126 |       0.233 |
| Sentence Embeddings + Nearest Centroid | **47.9%** | **0.330** |       0.460 |

The embedding classifier improved measured holdout accuracy by:

* **20.8 percentage points** over TF-IDF;
* **12.5 percentage points** over the majority baseline.

The result is still modest: **25 of 48 holdout examples were misclassified**.

### Evaluation Limitation

The holdout is small and imbalanced. Several intents have only one or two test examples. These numbers therefore describe this experimental split, not population-level performance.

###Confusion Matrix

A confusion matrix was generated from the 48-example intent holdout to visualize the model's intent-level errors.

Artifact:

```text
results/confusion_matrix.png
```
 
## 8. Historical-Resolution Retrieval

A FAISS index was built over 50,000 sampled historical AmazonHelp customer messages with their real historical responses.

Artifacts:

```text
data/processed/amazonhelp_faiss.index
data/processed/amazonhelp_retrieval_metadata.csv
```

Measured retrieval statistics:

| Metric                   | Result |
| ------------------------ | -----: |
| Average top-1 similarity |  0.805 |
| Average top-5 similarity |  0.738 |

The retrieval layer is intended to ground generated responses in how AmazonHelp historically handled similar cases.

Similarity is **not** treated as a response-quality score. It only measures semantic closeness between the incoming query and retrieved historical cases.

---

## 9. Escalation Decision

The final escalation layer uses transparent rules for strong signals such as:

* fraud/security concerns;
* explicit requests for a manager or human;
* investigation/escalation requests;
* unresolved support interactions;
* repeated contact or repeated failure;
* repeated/serious delivery failures.

Measured on the 191-example golden set:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        |  65.4% |
| HUMAN precision |  42.3% |
| HUMAN recall    |  17.7% |
| HUMAN F1        |  0.250 |

The low HUMAN recall is a known limitation.

A learned TF-IDF escalation classifier was also evaluated, but it achieved **0% HUMAN recall** on its 48-example holdout and was therefore not selected as the final escalation component.

The rules are intentionally conservative and should not be interpreted as a production-safe escalation policy given the low HUMAN recall and low escalation-label consistency.

---

## 10. Response Generation

The response-generation flow is:

1. classify the customer message;
2. retrieve top historical AmazonHelp cases;
3. provide the customer message plus historical evidence to Gemini;
4. generate a concise customer-facing draft;
5. prohibit unsupported refunds, guarantees, delivery promises, or invented actions.

When the Gemini Free Tier quota is exhausted, the agent uses a conservative local fallback rather than inventing a specific resolution.

### Response-Quality Evaluation and LLM-as-Judge

The intended response-quality rubric evaluates:

1. **Relevance** — Does the response address the customer's actual issue?
2. **Groundedness** — Is the response supported by the retrieved historical resolution?
3. **Safety** — Does it avoid unsupported promises, invented actions, or misleading claims?
4. **Conciseness** — Is it clear and appropriately brief?
5. **Overall quality** — Overall customer-facing response quality on a 1–5 scale.

A 10-example response-quality evaluation sample was prepared containing:

* customer message;
* historical resolution evidence;
* drafted response;
* fields for LLM-judge scoring;
* fields for human review.

The evaluation was attempted using multiple approaches:

* **Gemini:** successfully used during initial response-generation testing, but the available Free Tier quota was exhausted during development;
* **Qwen2.5-0.5B:** tested locally, but the generated judge output was not reliably parseable into the required structured scores;
* **FLAN-T5-small:** also tested locally, but the outputs were not reliable enough to establish valid structured judge scores.

Artifacts produced:

```text
results/local_llm_judge_results.csv
results/human_response_quality_review.csv
```

### Important limitation

A valid final LLM-as-judge score was **not established**.

Therefore:

* no LLM-as-judge response-quality score is reported;
* no LLM-vs-human agreement percentage is claimed;
* no response-quality metric is fabricated.

The human review file contains the 10 prepared responses and can be used to complete a judge-vs-human agreement study when a reliable judge model/API is available.

---

## 11. Failure Analysis

The embedding classifier produced 25 errors on the 48-example holdout.

### Failure Mode 1 — Product/service information vs account/security

`Product_or_Service_Information → Account_Login_or_Security` occurred 3 times.

**Hypothesis:** generic support/account wording in short messages can dominate the semantic representation and pull broad service questions toward the security/account class.

### Failure Mode 2 — Delivery tracking vs account/security

`Order_Delivery_Tracking → Account_Login_or_Security` occurred 2 times.

**Hypothesis:** short delivery messages contain little context, allowing generic account/support vocabulary to outweigh delivery semantics.

### Failure Mode 3 — Missing/wrong/damaged vs returns/replacements

`Missing_Wrong_or_Damaged_Item → Returns_Refunds_or_Replacements` occurred 2 times.

**Hypothesis:** these categories genuinely overlap because a wrong or damaged item often leads to a return or replacement.

### Failure Mode 4 — Delivery tracking vs delivery delay

Both directions occurred:

* `Delivery_Delay_or_Missed_Date → Order_Delivery_Tracking`
* `Order_Delivery_Tracking → Delivery_Delay_or_Missed_Date`

**Hypothesis:** short messages often do not make the distinction explicit. “Where is my package?” and “why hasn't it shipped?” are semantically close but correspond to different frozen intents.

### Failure Mode 5 — Broad language and sparse specialist classes

Additional errors included:

* `Missing_Wrong_or_Damaged_Item → Product_or_Service_Information`
* `Delivery_Delay_or_Missed_Date → Product_or_Service_Information`
* `Order_Cancellation_or_Change → Payment_or_Charge_Issue`
* `Payment_or_Charge_Issue → Prime_Video_Content_Availability`

**Hypothesis:** `Product_or_Service_Information` is the largest class, while several specialized intents have very few training examples. This makes the nearest-centroid representation biased toward broad or better-represented classes.

---

## 12. Human Consistency

A second-pass re-check was performed on 20 randomly selected golden examples.

| Measure              | Result |
| -------------------- | -----: |
| Intent agreement     |    85% |
| Escalation agreement |    35% |
| Overall agreement    |    30% |

This is **intra-annotator consistency**, not independent multi-annotator agreement.

The results suggest that the intent taxonomy was applied much more consistently than the escalation labels. Escalation conclusions should therefore be interpreted cautiously.

This human consistency check should **not** be confused with LLM-judge-vs-human agreement; that measurement was not successfully established.

---

## 13. What Is Misleading About My Headline Number?

The **47.9% intent accuracy** is useful as a benchmark, but it is easy to overinterpret.

It is measured on only **48 held-out examples**. Several intents have only one or two test examples. The classifier is also a simple nearest-centroid model rather than a fine-tuned production classifier.

The taxonomy itself contains genuinely confusable boundaries, especially:

* tracking vs delay;
* missing/damaged vs returns/replacements;
* broad service information vs specialized intents.

Therefore, 47.9% should be read as **a measured result on this particular experimental split**, not as a claim that approximately half of arbitrary Amazon support messages will always be classified correctly.

Similarly, retrieval similarity of 0.805 should not be interpreted as “80.5% response quality.” It is only an embedding-similarity measurement.

---

## 14. What I Did Not Build

To keep the project focused, I did not build:

* production cloud deployment;
* authentication or customer-account integration;
* production ticketing integration;
* fine-tuning on a large labeled corpus;
* advanced multi-stage RAG or reranking;
* production UI;
* automated policy verification;
* a completed, validated LLM-as-judge benchmark with judge/human agreement.

The focus remained on the core loop:

**classification → historical retrieval → escalation → grounded response**

---

## 15. One-Week Future Work

With one additional week, I would prioritize:

1. Expand and rebalance the golden set, especially minority intents.
2. Replace nearest-centroid classification with a supervised embedding classifier or lightweight fine-tuned model.
3. Add calibrated confidence thresholds and an abstain/needs-review state.
4. Improve delivery-intent separation with hierarchical classification.
5. Filter historical retrieval by predicted intent before response generation.
6. Add a reranker over the retrieved historical cases.
7. Complete the LLM-as-judge rubric and validate judge/human agreement using a reliable judge model.
8. Re-label escalation with a second independent annotator.
9. Add PII and policy safeguards before production deployment.
10. Build a small Streamlit demo for live evaluation.

---

## 16. Decision Log

1. **Selected AmazonHelp** because it offered a large directly linked support corpus and broad customer-message coverage among the leading candidates examined.

2. **Reconstructed customer → support response pairs** from thread links so that historical resolutions were preserved.

3. **Used a 10-intent taxonomy** to keep the system small while covering the major observed themes.

4. **Froze the taxonomy before final evaluation** to avoid changing labels after seeing model results.

5. **Excluded nine incomplete golden candidates rather than guessing** to reduce label noise.

6. **Used AI/rules only as pre-label assistance**; final labels were human-verified.

7. **Used the same train/test split across intent baselines** so results are comparable.

8. **Used a majority baseline** as the trivial reference.

9. **Used TF-IDF + Logistic Regression** as the simple interpretable baseline.

10. **Used sentence embeddings + nearest-centroid** to capture semantic similarity without expensive fine-tuning.

11. **Built a FAISS historical-resolution index** to ground response generation in real AmazonHelp examples.

12. **Selected transparent escalation rules** after the learned escalation model produced 0% HUMAN recall.

13. **Added a conservative response fallback** when Gemini quota is exhausted rather than inventing a specific resolution.

14. **Separated retrieval similarity from response quality**; similarity is not presented as proof of a good answer.

15. **Documented the missing LLM-as-judge result rather than fabricating a metric.**

---

## 17. Reproduction

From the project root:

```powershell
# Activate the project virtual environment.
.\.venv\Scripts\Activate.ps1

# Verify core packages used by the evaluation pipeline.
python -c "import pandas, numpy, sklearn, sentence_transformers, faiss; print('CORE PACKAGES OK')"

# Re-run the baseline comparison.
python .\src\22_baseline_comparison.py

# Re-run escalation evaluation.
python .\src\18_escalation_rules.py

# Regenerate consolidated report metrics.
python .\src\27_generate_report_metrics.py
```

Existing generated artifacts are stored under:

```text
data\processed\
data\golden\
results\
```

For a fresh checkout, place the raw TWCS dataset at:

```text
data\raw\twcs.csv
```

The raw dataset and API credentials are intentionally not committed.

The local `.env` file is used for development credentials but is excluded from Git through `.gitignore`.

---

## 18. Project Structure

```text
hiver-sde-agent/
│
├── README.md
├── .gitignore
|
├── app/
│   └── streamlit_app.py
│
|
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── results/
│
└── src/
    ├── 01_brand_analysis.py
    ├── 02_brand_conversation_analysis.py
    ├── 03_brand_issue_analysis.py
    ├── 04_resolution_analysis.py
    ├── 05_build_conversations.py
    ├── 06_validate_conversations.py
    ├── 07_intent_discovery.py
    ├── 08_create_intent_taxonomy.py
    ├── 09_prepare_golden_candidates.py
    ├── 10_ai_assisted_golden_labeling.py
    ├── 11_fast_human_labeling.py
    ├── 12_finalize_golden_set.py
    ├── 13_train_intent_baseline.py
    ├── 14_embedding_intent_classifier.py
    ├── 15_build_resolution_retriever.py
    ├── 16_response_generator.py
    ├── 17_escalation_classifier.py
    ├── 18_escalation_rules.py
    ├── 19_analyze_escalation_failures.py
    ├── 20_support_agent.py
    ├── 21_evaluation_harness.py
    ├── 22_baseline_comparison.py
    ├── 23_response_evaluation.py
    ├── 24_failure_analysis.py
    ├── 25_show_failure_pairs.py
    ├── 26_human_agreement_check.py
    ├── 27_generate_report_metrics.py
    ├── 28_local_llm_judge.py
    └── 29_confusion_matrix.py

Large/raw/generated artifacts are intentionally excluded from Git through `.gitignore`.

---

## 19. Final Status

The project has:

* a reconstructed AmazonHelp support corpus;
* a frozen 10-intent taxonomy;
* a 191-example human-verified golden set;
* majority and TF-IDF baselines;
* a semantic embedding classifier;
* FAISS historical-resolution retrieval;
* an explicit escalation layer;
* an end-to-end support-agent pipeline;
* automated evaluation metrics;
* failure analysis;
* human consistency evidence;
* an LLM-as-judge rubric and evaluation artifacts;
* a documented decision log.

### Known Evaluation Limitation

A validated LLM-as-judge score and LLM-judge-vs-human agreement percentage were **not established**.

Gemini API quota was exhausted during development, while local Qwen and FLAN-T5 experiments did not produce reliable structured judge outputs.

No unsupported response-quality score is reported.

The project therefore prioritizes transparent evidence over an artificial headline metric.
    
