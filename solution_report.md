# Part 4: AI Solution Design for a Business Problem
## AI-Powered Customer Support Ticket Routing and Sentiment Resolution

---

## Task 1: Business Domain

**Domain Selected: Customer Support**

Customer support is one of the highest-volume, most operationally expensive functions in any product or service business. Organisations receive thousands of tickets per month across channels — chat, email, phone, and social media — and must triage, route, and resolve each one efficiently. As the monthly metrics dataset shows, manual processing of 2,500–3,150 tickets per month consumes 330–567 labour hours, carries error rates up to 11%, and produces inconsistent customer satisfaction scores ranging from 6.4 to 8.6 out of 10.

This variability and inefficiency makes customer support an ideal domain for AI-driven optimisation.

---

## Task 2: Business Problem Definition

### What problem is being solved?

Customer support teams receive a high volume of incoming tickets across multiple channels (chat, email, phone, social media). Each ticket must be:

1. Read and understood by a human agent
2. Classified by urgency and sentiment
3. Routed to the correct team or specialist
4. Responded to, either via a template or a custom reply
5. Followed up until resolved

This process is entirely manual, slow, error-prone, and inconsistent. The data confirms this: resolution times average 18–44 hours, error rates peak at 11%, and 33% of tickets are flagged as urgent yet still wait in the same queue as non-urgent ones.

The business problem is:

> **How can AI automatically classify the sentiment and urgency of incoming customer support tickets and route them to the right team or auto-resolution path — reducing manual workload, resolution time, and error rate while improving customer satisfaction?**

### Who are the users and stakeholders?

| Stakeholder | Role |
|-------------|------|
| **Support Agents** | Directly use the system; receive pre-triaged, pre-labelled tickets |
| **Support Managers** | Monitor queue health, SLA compliance, escalation rates |
| **Customers** | End beneficiaries — faster, more accurate responses |
| **IT / MLOps Team** | Deploy and maintain the AI pipeline |
| **Compliance / Legal** | Ensure customer data privacy (GDPR, PDPA) |
| **Product Team** | Use ticket insights to identify product issues |

### What is the current manual process?

1. A customer submits a ticket via any channel
2. An agent reads the ticket and decides urgency manually
3. The ticket is categorised by hand (billing, technical, general enquiry, etc.)
4. The agent routes it to the appropriate team manually
5. The team responds — either with a template or a custom reply
6. Resolution is confirmed, and the case is closed

This process involves no automation. Every step requires human reading, judgement, and action.

### Limitations of the current process

| Limitation | Business Impact |
|------------|----------------|
| No automatic urgency detection | Critical issues sit in the same queue as routine queries |
| Manual categorisation is inconsistent | Different agents classify the same ticket differently |
| No sentiment analysis | Frustrated customers are not prioritised |
| High manual processing hours (330–567/month) | High staffing cost, agent burnout |
| Average resolution time 18–45 hours | Poor customer experience, low CSAT |
| Error rate 4–11% | Misroutes, double-handling, repeat contacts |
| No feedback loop | Mistakes are not captured to improve the process |

---

## Task 3: AI Task Type

### Selected AI Task Type: **Text Classification** (with Sentiment Analysis as a sub-task)

**Why this is suitable:**

The core problem is categorising free-text messages into a known set of classes — sentiment labels (positive, neutral, negative), urgency flags, and topic clusters. This is exactly what text classification models are designed to do.

The justification for choosing text classification over alternatives:

- **Not image classification** — the data is text (messages, emails, transcripts)
- **Not regression** — the output is a discrete class label, not a continuous value
- **Not anomaly detection** — most tickets are normal; we are classifying all of them, not finding outliers
- **Not sequence prediction** — we are not forecasting future ticket volume; we are classifying individual messages at inference time
- **Text classification is the right fit** — each incoming message is an independent unit that needs a label assigned to it before routing action is taken

Within the text classification umbrella, the solution also incorporates:
- **Sentiment analysis** to detect customer emotion and frustration level
- **Topic/intent classification** to determine what the ticket is about
- **Urgency scoring** to decide routing priority

---

## Task 4: Data Requirement Plan

### Type of data needed

| Data Type | Description |
|-----------|-------------|
| **Primary (unstructured)** | Raw customer ticket text — chat messages, emails, transcripts |
| **Secondary (structured)** | Ticket metadata — channel, timestamp, word count, urgent flag |
| **Historical labels** | Past tickets with agent-assigned sentiment and routing decisions |
| **Outcome data** | Resolution time, CSAT score, re-open rate per ticket |

### Structured vs. Unstructured

- **Unstructured**: The raw customer message text (primary input to the NLP model)
- **Structured**: Channel type, word count, urgent flag, month, resolution time, error rate, CSAT score (used for monitoring and evaluation)

### Input Features

| Feature | Type | Role |
|---------|------|------|
| `customer_message` | Text (unstructured) | Primary model input |
| `channel` | Categorical | Auxiliary feature (chat vs email vs phone) |
| `word_count` | Numeric | Proxy for ticket complexity |
| `urgent_flag` | Binary | Validated urgency signal |
| `time_of_day` | Numeric | Routing priority context |
| `TF-IDF vectors` | Numeric (sparse) | Engineered text features |
| `Embedding sequences` | Numeric (dense) | Deep text representation for LSTM |

### Target Variable / Labels

| Target | Type | Values |
|--------|------|--------|
| `sentiment_label` | Categorical (3-class) | positive, neutral, negative |
| `urgency_score` | Binary or ordinal | 0 (routine), 1 (urgent) |
| `topic_cluster` | Categorical (multi-class) | billing, technical, general, refund, etc. |

### Data Collection Method

1. **Historical export**: Pull 12+ months of resolved tickets from the CRM/helpdesk system (e.g. Zendesk, Freshdesk)
2. **Agent labelling**: Have agents retrospectively label a sample of 1,000–2,000 tickets with correct sentiment and topic (used as gold-standard training data)
3. **Automated label mining**: Use resolution outcome (ticket closed with template = likely routine; escalated to senior agent = likely negative/urgent) as a weak signal for additional label generation
4. **Ongoing collection**: After deployment, capture agent corrections as new labelled training data

### Data Quality Risks

| Risk | Mitigation |
|------|-----------|
| **Duplicate / near-duplicate messages** | Deduplicate during preprocessing; use diverse sampling for training |
| **Class imbalance** (few positive, many neutral) | Oversample minority classes or use class-weighted loss |
| **Inconsistent agent labels** (two agents label same ticket differently) | Inter-annotator agreement checks; majority vote labelling |
| **Missing data** (some fields blank) | Impute or exclude during training; flag for retraining triggers |
| **Language/dialect variation** | Normalise slang, abbreviations; consider multilingual embeddings if needed |
| **Biased historical labels** (certain demographics misrouted historically) | Fairness audit on training data by channel, language, geography |

---

## Task 5: Model Recommendation

### Recommended Model: **Bidirectional LSTM (BiLSTM)**

With an optional upgrade to a **fine-tuned BERT Transformer** for production.

### Architecture

```
Input: Tokenised, padded customer message (sequence length = 20–50)
    │
    ▼
Embedding Layer (vocab size × 64 dimensions)
    Converts integer token IDs into dense semantic vectors.
    Trainable from scratch given sufficient data.
    │
    ▼
Bidirectional LSTM (64 units per direction = 128 total)
    Reads the sequence left-to-right AND right-to-left.
    Captures context from both directions —
    e.g. "not resolved" and "resolved not" are both understood.
    Returns sequences for every time step.
    │
    ▼
Global Max Pooling
    Selects the most salient signal across all time steps.
    Produces a fixed-length representation regardless of input length.
    │
    ▼
Dropout (rate = 0.3)
    Regularisation — randomly zeros 30% of neurons during training
    to prevent the model from memorising training examples.
    │
    ▼
Dense Layer (64 neurons, ReLU activation)
    Non-linear feature transformation.
    │
    ▼
Output Dense (3 neurons, Softmax activation)
    Produces probabilities: P(negative), P(neutral), P(positive)
    Highest probability wins → predicted class.

Loss function : Sparse Categorical Cross-Entropy
Optimizer     : Adam (lr = 0.001, with decay)
Epochs        : 10–20 with early stopping
Batch size    : 32
```

### Why BiLSTM over alternatives?

| Model | Why Not / Why Yes |
|-------|------------------|
| **Naive Bayes / Logistic Regression** | Good baselines (used in Part 3) but cannot capture word order or long-range context |
| **Standard LSTM** | Only reads left-to-right; misses right-to-left context ("issue not resolved" vs "resolved issue not") |
| **BiLSTM** | **Chosen** — reads both directions, handles variable-length text, works well on 100–10,000 training examples |
| **Transformer (BERT)** | Superior but requires much larger data (10k+ examples) and significant compute; recommended as a future upgrade once sufficient data is collected |
| **CNN for text** | Captures local patterns (n-grams) but not long-range dependencies |

### Upgrade Path

Once the labelled dataset grows to 10,000+ records and compute budget allows:

```
Fine-tune DistilBERT or RoBERTa on the support ticket classification task.
Pre-trained transformer models already understand language context;
fine-tuning adapts them to the support domain in far fewer training steps.
```

---

## Task 6: Evaluation Plan

### Technical Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | Correct predictions / Total predictions | > 90% |
| **Macro F1-score** | Harmonic mean of precision & recall (equal class weight) | > 0.88 |
| **Precision** (negative class) | True negatives / All predicted negative | > 0.90 |
| **Recall** (negative class) | True negatives / All actual negative | > 0.92 |
| **AUC-ROC** | Area under receiver operating curve | > 0.95 |
| **Confidence calibration** | Predicted probability vs actual frequency | Well-calibrated |

Negative class precision and recall are prioritised because misclassifying a frustrated, negative customer as neutral is the most damaging error for business outcomes.

### Business Metrics

| Metric | Current Baseline | Target (6 months post-deployment) |
|--------|-----------------|----------------------------------|
| Manual processing hours / month | 330–567 hours | < 200 hours (-55%) |
| Average resolution time | 18–44 hours | < 10 hours (-70%) |
| Error rate | 4–11% | < 3% (-65%) |
| CSAT score | 6.4–8.6 / 10 | > 8.0 consistently |
| Monthly cases handled | 2,466–3,154 | 5,000+ (scale without adding headcount) |
| Agent time on complex issues | ~40% of time | > 75% of time |

### Possible Failure Cases

| Failure Mode | Description | Mitigation |
|--------------|-------------|-----------|
| **Misclassified urgency** | Urgent ticket classified as routine; customer waits too long | Confidence threshold: route to agent review if P < 0.80 |
| **Sarcasm / irony** | "Oh great, another billing error" classified as positive | Include adversarial examples in training; agent correction loop |
| **Domain shift** | New product launches generate out-of-vocabulary ticket types | Monthly monitoring of unknown token rate; trigger retraining |
| **Language mismatch** | Non-English tickets misclassified | Language detection gate; multilingual model or separate pipeline |
| **Template gaming** | Customers learn to phrase tickets to get faster service | Monitor distribution shift; route based on account history too |

### Human Review and Validation Process

1. **Confidence threshold gate**: Any prediction with softmax probability < 0.80 is flagged for human review before routing
2. **Weekly spot-check**: Support manager reviews a random sample of 50 auto-routed tickets to assess correctness
3. **Agent feedback button**: Agents can mark any AI routing as incorrect in one click; this data feeds back into the training pipeline
4. **Monthly model audit**: MLOps team reviews confusion matrix, class distribution of predictions, and drift metrics monthly
5. **Quarterly fairness review**: Ensure routing rates are equitable across customer demographics and channels

---

## Task 7: Responsible AI Considerations

### Bias in Data

The historical training data reflects the decisions of past agents, who may have themselves been inconsistent or biased. For example:

- Tickets submitted via social media may have been historically deprioritised versus email, even if they were equally urgent
- If agents consistently misrouted complaints from non-English-speaking customers, the model will learn this pattern as correct behaviour
- Templates in the dataset may over-represent certain types of issues and under-represent edge cases

**Mitigation**: Conduct a bias audit on training data stratified by channel, language, and ticket type. Use balanced sampling. Monitor routing outcomes by subgroup post-deployment.

### Incorrect Predictions

No model is perfect. A misclassified ticket can delay resolution for a customer who is already frustrated. False negatives (missing urgent tickets) are especially harmful because they delay service for people who need it most.

**Mitigation**: Use the confidence threshold gate (Task 6). Never fully automate final action without a human check for low-confidence predictions. Implement SLA alerts that fire independently of AI classification.

### Privacy Concerns

Customer support messages often contain personally identifiable information (PII) — names, account numbers, addresses, medical details, complaint specifics. Training a model on this data raises data protection obligations under GDPR (EU), PDPA (India/Singapore), and equivalent legislation.

**Mitigation**:
- Anonymise or pseudonymise customer PII before use in training
- Store training data in a secure, access-controlled environment
- Obtain explicit data governance approval before using ticket data for model training
- Do not use ticket text data for any purpose other than the stated AI task
- Implement data retention policies; delete raw ticket text from ML systems after model training

### Over-reliance on AI

If agents trust the AI routing completely without applying their own judgement, errors will be amplified rather than caught. The system risk is that agents stop reading tickets carefully and simply act on the AI's suggestion.

**Mitigation**: Train agents on the purpose and limitations of the AI system. Emphasise that AI is a recommendation, not a command. Show confidence scores alongside predictions so agents can assess reliability. Maintain a feedback channel for agents to correct the AI.

### Impact on Users (Customers)

Customers may receive automated, impersonal responses when they expected human empathy — particularly in emotionally charged situations (complaints, billing disputes, service failures).

**Mitigation**: The AI system routes tickets; it does not generate responses autonomously. Human agents still write and approve all customer-facing communication. Auto-replies (template responses) are used only for clearly neutral, routine enquiries where the customer's need is unambiguous.

### Need for Human Oversight

AI classification should be treated as a first-pass triage tool, not a final decision system. The following decisions must always involve a human:

- Closing a ticket without agent review
- Issuing a refund or credit based on ticket content
- Escalating to legal or compliance
- Engaging media or VIP customers

### Summary of Responsible AI Safeguards

| Risk | Safeguard |
|------|-----------|
| Incorrect routing | Confidence threshold + human review gate |
| Bias in training data | Stratified audit, balanced sampling |
| PII in training data | Anonymisation before training |
| Agent over-trust in AI | Training, confidence display, feedback loop |
| Impersonal customer experience | AI routes only; humans respond |
| Model drift over time | Monthly monitoring, quarterly retraining |
| Accountability gap | Full audit trail of every AI routing decision |

---

## Task 8: Final Solution Summary

---

### One-Page Solution Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│              AI SOLUTION DESIGN — EXECUTIVE SUMMARY                  │
│         Customer Support Ticket Routing & Sentiment Analysis         │
└─────────────────────────────────────────────────────────────────────┘

BUSINESS PROBLEM
────────────────
Customer support teams process 2,500–3,150 tickets/month manually.
Average resolution time is 18–44 hours. Error rate reaches 11%.
CSAT fluctuates between 6.4 and 8.6/10 with no consistent trend.
Manual triage cannot scale as case volume grows.

PROPOSED AI SOLUTION
─────────────────────
A Bidirectional LSTM text classification model that:
  1. Ingests incoming ticket text from any channel (chat, email, phone, social)
  2. Classifies sentiment (positive / neutral / negative)
  3. Detects urgency and topic category
  4. Routes to the correct team or auto-resolution path
  5. Learns continuously from agent corrections and CSAT outcomes

REQUIRED DATA
─────────────
  • 1,500+ labelled historical support tickets (already available)
  • Agent routing decision labels (training signal)
  • Customer satisfaction scores (outcome signal for retraining)
  • Channel metadata, timestamps, word count, urgent flag

MODEL RECOMMENDATION
─────────────────────
  Architecture : Bidirectional LSTM
  Input        : Tokenised, padded text sequences (length 20–50)
  Layers       : Embedding (64-dim) → BiLSTM (128 units) →
                 GlobalMaxPool → Dropout(0.3) → Dense(64) → Softmax(3)
  Loss         : Sparse Categorical Cross-Entropy
  Optimizer    : Adam (lr = 0.001)
  Upgrade path : Fine-tune DistilBERT once dataset > 10,000 records

EXPECTED BUSINESS IMPACT (6-month target)
──────────────────────────────────────────
  Manual processing hours    : 567 → < 200/month   (-65%)
  Average resolution time    : 35h  → < 10h         (-71%)
  Error rate                 : 11%  → < 3%           (-73%)
  CSAT score                 : 6.7  → > 8.0          (+19%)
  Cases handled / month      : 2,800 → 5,000+        (+78%)

RISKS & MITIGATION PLAN
────────────────────────
  ┌──────────────────────────────┬───────────────────────────────────┐
  │ Risk                         │ Mitigation                        │
  ├──────────────────────────────┼───────────────────────────────────┤
  │ Misclassified urgent tickets │ Confidence threshold < 0.80 →     │
  │                              │ route to human review             │
  ├──────────────────────────────┼───────────────────────────────────┤
  │ Bias in historical labels    │ Stratified audit + balanced        │
  │                              │ sampling before training          │
  ├──────────────────────────────┼───────────────────────────────────┤
  │ Customer PII in training     │ Anonymise data before training;   │
  │ data                         │ GDPR-compliant storage            │
  ├──────────────────────────────┼───────────────────────────────────┤
  │ Agent over-reliance on AI    │ Training, confidence display,     │
  │                              │ one-click correction feedback     │
  ├──────────────────────────────┼───────────────────────────────────┤
  │ Model drift (new products)   │ Monthly drift monitoring,         │
  │                              │ quarterly retraining pipeline     │
  └──────────────────────────────┴───────────────────────────────────┘

IMPLEMENTATION TIMELINE
────────────────────────
  Month 1: Data collection, cleaning, labelling
  Month 2: Model training, evaluation, threshold tuning
  Month 3: Pilot deployment (10% of tickets, shadow mode)
  Month 4: Full deployment with agent feedback loop live
  Month 5: First model retraining on collected feedback
  Month 6: Business impact review against targets above
```

---

*Report prepared for: AI Business Analyst Mini Project — Part 4*
*Domain: Customer Support | Model: Bidirectional LSTM | Data: 12-month operational metrics*
