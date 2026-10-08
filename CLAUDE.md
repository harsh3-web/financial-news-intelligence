# Financial News Intelligence — Project Context & Mentoring Handoff

> This file is the handoff from a mentoring conversation in the Claude app.
> **Claude: read this whole file before responding.** Continue the mentoring exactly where it stopped (see "NEXT STEP" at the bottom), following the teaching rules below.

---

## 1. Who I am and what this project is for

- **Student:** Harsh Sharma, B.Tech + M.Tech dual degree, Civil Engineering, IIT Bhubaneswar.
- **Target:** ₹20+ LPA Data Scientist / ML Engineer / AI roles.
- **Already studied:** Statistics, ML, ANN, CNN, RNN, NLP, Transformers (theory).
- **Previous projects:** Movie Sentiment Analyzer (Embeddings + SimpleRNN, Keras), YOLO road defect detection, recommendation system, churn prediction, data analytics internship.
- **NOT in scope:** RAG and Agentic AI (learning separately). Do not add them to this project.
- **Interview weakness to train:** previously rejected partly due to nervousness and not communicating clearly. Train calm, structured answers using **What → Why → How → Example → Trade-off**.

**Project:** *Financial News Intelligence — Comparative NLP System for Financial Sentiment Analysis.*
Classify financial news sentences as **positive / neutral / negative**, and compare:

1. TF-IDF + Logistic Regression
2. SimpleRNN
3. LSTM
4. GRU
5. FinBERT (fine-tuned by us)

Compared on: accuracy, precision, recall, **Macro-F1 (primary)**, confusion matrix, parameters, training time, inference latency, complexity, error analysis.
**Never invent results. All numbers must come from experiments actually run.**

---

## 2. Teaching rules (follow strictly)

For every important concept: **WHAT it is → WHY we need it → HOW it works → small example → code → code explanation → expected output → small task for me.**

1. Don't dump everything at once. One step per message; wait for me to finish each major stage.
2. Explain before coding. Beginner-friendly language, even though I know the theory.
3. Don't assume I know implementation details.
4. Give **complete, runnable, copy-pasteable code** for the current step (I strongly prefer full code blocks over partial snippets). Explain important lines.
5. Don't rename variables once established (see Section 6).
6. Human-readable comments. No unnecessarily complicated frameworks.
7. Never fabricate results. Never skip data leakage or evaluation.
8. Give exercises so I build it myself. Review my answers like a tough but fair interviewer: point out vague reasoning, arithmetic slips, and claims without examples.
9. If several approaches are valid, explain trade-offs and recommend one.
10. End every major phase with a short checklist of what I must understand.
11. If I ask something off-roadmap, answer it, then bring me back.
12. Optimize for technical depth and interview value, not GitHub looks. Keep it finishable.

**Observed patterns in my answers (coach on these):**
- I tend to stop examples before the punchline: always push me to **finish examples with actual numbers**.
- I sometimes make arithmetic slips (e.g. proportions summing to 80%, precision/recall miscalculated). Encourage sanity checks.
- My reasoning is sometimes vague ("the model will get to know..."). Push for concrete, specific reasons.
- I'm good at spotting data issues by eye (found the encoding corruption, questioned a debatable label).

---

## 3. Roadmap (current position marked)

| Phase | Topic | Status |
|---|---|---|
| 1 | Problem formulation, research questions, metrics, folder structure | ✅ Done |
| 2 | Dataset + EDA → `notebooks/01_data_exploration.ipynb` | 🟡 **In progress (Steps 2.1–2.4 done)** |
| 3 | Preprocessing: cleaning, numbers, tokenization, stratified split, leakage; reusable pipeline in `src/` | ⏳ |
| 4 | TF-IDF + Logistic Regression baseline (BoW, TF, IDF math, Macro-F1, confusion matrix); save results | ⏳ |
| 5 | SimpleRNN (tokenizer, vocab, padding, embeddings, hidden state, BPTT, vanishing gradients; tiny manual example first) | ⏳ |
| 6 | LSTM (gates, cell state, equations); compare with SimpleRNN | ⏳ |
| 7 | GRU (update/reset gates); compare SimpleRNN vs LSTM vs GRU | ⏳ |
| 8 | Transformer theory (self-attention with a real small sentence, Q/K/V, scaled dot-product, multi-head, positional encoding, residuals, layer norm, encoder/decoder types, BERT, pretraining vs fine-tuning) | ⏳ |
| 9 | FinBERT fine-tuning (tokenizer, input IDs, attention masks, special tokens, classification head, freezing, LR, batch size, epochs, weight decay, warmup, early stopping) | ⏳ |
| 10 | Model comparison table (only real numbers) + trade-off analysis | ⏳ |
| 11 | Error analysis (negation, numbers, comparatives, mixed sentiment, domain terms, ambiguity) across LR / LSTM / GRU / FinBERT — no cherry-picking | ⏳ |
| 12 | OPTIONAL: news → market movement (look-ahead bias, time-based split). Never claim sentiment causes price moves | ⏳ |
| 13 | Streamlit app (sentiment + confidence + per-class probabilities), later FastAPI | ⏳ |
| 14 | MLOps: Git, model saving, MLflow, FastAPI, Docker, monitoring, versioning — only after core works | ⏳ |
| 15 | README, architecture diagram, results, limitations, 2–3 honest resume bullets | ⏳ |
| 16 | Tough mock interviews with follow-ups, What→Why→How→Example→Trade-off | ⏳ |

---

## 4. Environment

- **OS / editor:** Windows, VS Code, PowerShell terminal.
- **Project root:** `H:\Desktop\financial-news-intelligence` (Git initialized, branch `main`).
- **Virtual env:** `.venv` (activate: `.venv\Scripts\activate`). Python 3.13.
- **Installed so far:** pandas 3.0.6, numpy, matplotlib, seaborn, scikit-learn 1.9.1, ipykernel. Recorded with `pip freeze > requirements.txt` (rerun after each new install).
- **pandas 3 note:** text columns show dtype `str`, not `object`.
- **GPU plan:** Phases 2–7 run locally on CPU. Phase 9 (FinBERT) runs in a VS Code notebook connected to a **Google Colab GPU kernel** via the official Google Colab VS Code extension. The Colab kernel can't see local files: data must be cloned from GitHub or uploaded, and the trained model downloaded back.
- **Latency rule:** all inference latency measured on the **same local CPU, batch size 1**, for every model.

### Planned frameworks
- scikit-learn: TF-IDF + LR
- TensorFlow/Keras: SimpleRNN / LSTM / GRU (familiar from my earlier project)
- Hugging Face Transformers (PyTorch): FinBERT
- One shared `evaluate()` function (scikit-learn metrics) for all models.

---

## 5. Folder structure

```
financial-news-intelligence/
├── CLAUDE.md                 # this file
├── problem_statement.md      # my problem statement (Phase 1)
├── setup_project.py
├── README.md, requirements.txt, .gitignore   # .gitignore covers venv/, .venv/, models/, caches
├── data/
│   ├── raw/FinancialPhraseBank-v1.0/   # original files, NEVER edited
│   └── processed/                      # cleaned data + saved splits (Phase 3)
├── notebooks/01_data_exploration.ipynb
├── src/__init__.py                     # reusable code goes here
├── models/
├── results/{metrics,figures,error_analysis}/
├── app/
└── tests/
```

Rules: `data/raw/` is read-only. Any function used in more than one notebook moves into `src/`.

---

## 6. Key decisions and established names

### Decisions
- **Dataset:** Financial PhraseBank, **`Sentences_75Agree.txt`** (3,453 sentences). Not AllAgree (too easy, hides model differences), not 50Agree (noisy labels).
- **Never mix agreement subsets** across train/test: they are nested (AllAgree ⊂ 75Agree ⊂ 66Agree ⊂ 50Agree) → leakage.
- **Primary metric:** Macro-F1. Secondary: accuracy, per-class precision/recall (esp. **negative recall**). Diagnostic: confusion matrix. Cost: params, training time, inference latency.
- **Fairness rules:** same split for all models (created once, saved to disk), same evaluation code, fixed random seeds, test set used only once at the end (tune on validation), same hardware for latency.
- **FinBERT leakage trap:** `ProsusAI/finbert` was already fine-tuned on Financial PhraseBank → evaluating it on our test set would be contaminated. Use a FinBERT checkpoint that is only **domain-pretrained** (not fine-tuned on PhraseBank) and fine-tune it ourselves on our train split. Verify the current checkpoint availability in Phase 9.
- **Class display order everywhere:** `["negative", "neutral", "positive"]`.

### Established variable/function names (do not rename)
`PROJECT_ROOT`, `DATA_FILE`, `FIGURES_DIR`, `load_phrasebank()`, `bad_lines`, `df` (columns `sentence`, `label`, `dup_key`), `normalize_for_dup_check()`, `class_order`, `label_counts`, `exact_dup_mask`, `near_dup_mask`, `non_ascii_mask`.

---

## 7. Dataset facts

- **Source:** official zip on Hugging Face: `https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip` (the old `load_dataset("financial_phrasebank")` script-based loading is unreliable with new `datasets` versions).
- **License:** CC BY-NC-SA 3.0 (non-commercial; state this in README).
- **Origin:** Malo et al. (2014). English financial news sentences, mostly about European (largely Finnish) companies. Each sentence labelled by 5–8 annotators **from an investor's viewpoint** (effect on stock price); financially irrelevant sentiment = neutral.
- **Format:** one line per sentence, `sentence@label`. Parsed with `line.rsplit("@", 1)` (split at the LAST `@`).
- **Limitations:** single sentences (no document context), old news, narrow geographic source, some genuinely ambiguous labels (e.g. "plans to double revenues" labelled positive: a plan, not a result).

---

## 8. EDA findings so far (data quality log)

- **Encoding:** file is Latin-1, not UTF-8 (UTF-8 decoding fails). Load with `encoding="latin-1"`.
- **Mojibake:** 74 sentences (2.1%) contain corrupted characters baked into the file, e.g. `Sepp+ñl+ñ` = Seppälä, `+àland` = Åland, `L+ñnnen`, `J+ñrvi-Suomen`, `Lemmink+ñinen`. Most common patterns: `+ñ` 43, `+à` 16, `+ó` 9, `+Â` 8, `+Ñ` 6, `-¦` 6, plus rarer `-á`, `+ä`, `-¼`, `+£`, `-®`, `-«`, `+®`, `-Á`, `+é`. Almost entirely in proper names (no sentiment words affected).
  - Hypothesis: original UTF-8 bytes were decoded as an old DOS code page (cp437) and saved as Latin-1, with unmappable box-drawing characters replaced by `+`/`-` (e.g. ä = UTF-8 `C3 A4` → cp437 `├ñ` → `+ñ`). **Must verify** before repairing; check repaired names against real companies (Bank of Åland, Lemminkäinen).
  - Decision: low priority, repair in Phase 3 without touching raw files.
- **Inconsistent entity spelling:** `Sepp+ñl+ñ` and `Seppala` both appear for the same company.
- **Pre-tokenized text:** spaces before punctuation (`Finland , plans`).
- **Broken numbers:** decimals sometimes split, e.g. `EUR10 .1 m` (= EUR 10.1 m). Handle carefully in Phase 3; do NOT blindly delete numbers.
- **Class distribution:** neutral 2,146 (62.1%), positive 887 (25.7%), negative 420 (12.2%). Figure saved: `results/figures/01_class_distribution.png` (fix title typo "Agress" → "Agree" and re-save).
- **Majority-class baseline (always-neutral):** accuracy 62.1%, Macro-F1 0.255 (neutral F1 = 2×0.621×1/1.621 = 0.766; ÷3). This is the floor every model must beat.
- **Exact duplicates:** 5 sentences duplicated once each (10 rows, 5 extra copies), all neutral, **0 conflicting labels**. Plan: drop extra copies before splitting.
- **Near-duplicates** (identical after lowercasing + numbers → `<num>`): 24 groups, 55 rows, **0 conflicting labels**. Largest are positive earnings templates, e.g. "operating profit rose to eur <num> mn from eur <num> mn in the corresponding period in <num>" (6 sentences). Plan: **group-aware stratified split** using `dup_key` so a template group never spans train and test (otherwise test scores are inflated by memorized templates). Limitation: this check only catches sentences differing in numbers, not in company names.

---

## 9. Code written so far (`notebooks/01_data_exploration.ipynb`)

- **Cell 1:** imports, `PROJECT_ROOT` (steps up from `notebooks/`), `DATA_FILE` path check.
- **Cell 2:** proves UTF-8 decoding fails on raw bytes.
- **Cell 3:** `load_phrasebank(file_path)` → returns `(df, bad_lines)`:
  ```python
  def load_phrasebank(file_path):
      """Read a Financial PhraseBank file into a DataFrame with 'sentence' and 'label' columns."""
      rows = []
      bad_lines = []
      with open(file_path, encoding="latin-1") as f:
          for line_number, line in enumerate(f, start=1):
              line = line.strip()
              if not line:
                  continue
              if "@" not in line:
                  bad_lines.append((line_number, line))
                  continue
              sentence, label = line.rsplit("@", 1)
              rows.append({"sentence": sentence.strip(), "label": label.strip()})
      return pd.DataFrame(rows), bad_lines
  ```
  Result: 3,453 rows, 0 bad lines, labels exactly `neutral/positive/negative`.
- **Cell 4:** label check, `df.info()`, Sepp lookup, `df.head()`.
- **Cell 5:** non-ASCII / mojibake pattern counter.
- **Cell 6:** class distribution table + bar chart saved to `results/figures/`.
- **Cell 7:** exact duplicates + conflicting-label check.
- **Cell 8:** `normalize_for_dup_check()` (lowercase, `\d+(?:[.,]\d+)*` → `<num>`, collapse spaces) → `df["dup_key"]`, near-duplicate groups.
- **Markdown cell:** data quality log (update with duplicate findings).

---

## 10. NEXT STEP

**Phase 2, Step 2.5: sentence length distribution, word frequencies, outliers.**

Teach and implement (one step at a time, with my tasks at the end):
- Sentence length in words (note: punctuation counts as tokens because the text is pre-tokenized): overall stats, by label, histogram + box plot by label, saved to `results/figures/`.
- Percentiles (95th / 99th) and how they will decide `max_len` for RNN padding (Phase 5) and FinBERT truncation (Phase 9).
- Shortest and longest sentences (outliers): inspect, don't delete blindly.
- Most frequent words overall and per class (before and after stopword removal), and why removing stopwords like "not", "down", "up", "below" would be harmful for financial sentiment.
- Then: finish `01_data_exploration.ipynb` with a summary cell, give the Phase 2 checklist, and only then start Phase 3.

Small pending items from me:
- Update the data quality log with the duplicate findings.
- Fix the plot title typo and re-save the figure.
- Commit: `git add .` → `git commit -m "Phase 2: loading, encoding, class distribution, duplicates"`.
