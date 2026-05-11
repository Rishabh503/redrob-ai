# Resume Matching Engine

Built for the Redrob AI Campus Hackathon — matches candidate resumes against job descriptions using TF-IDF and cosine similarity.

---

## What It Does

* Normalizes messy/typo-ridden resume skill data using a canonical alias map
* Computes TF-IDF vectors for each resume
* Reads Job Descriptions from an Excel file you fill in
* Ranks the Top 3 matching candidates per JD using cosine similarity

---

## Files

```
resume_matching_engine.py     ← main script
job_descriptions_input.xlsx   ← Excel file where you add your JDs
README.md                     ← this file
```

---

## Setup

You only need Python 3 and one library:

```bash
pip install openpyxl
```

No other external libraries needed — the engine uses only Python's built-in `math` module for all calculations.

---

## How to Test Against New JDs

### Step 1 — Open the Excel file

Open `job_descriptions_input.xlsx`. It looks like this:

| JD # | Company          | Role             | Required Skills               | Preferred Skills     |
| ---- | ---------------- | ---------------- | ----------------------------- | -------------------- |
| JD-1 | Kakao (Seoul)    | ML Engineer      | python, machine_learning, ... | nlp, bert, ...       |
| JD-2 | Naver (Seongnam) | Backend Engineer | java, spring_boot, ...        | rest_api, ci_cd, ... |

### Step 2 — Add your new JD as a new row

Just add a new row below the existing ones. Example:

| JD-4 | Google (Bangalore) | Data Engineer | python, sql, postgresql, docker | aws, kubernetes, ci_cd |

> **Important:** Skills must be written as canonical names (see the skill reference table below). Separate multiple skills with commas.

### Step 3 — Run the script

```bash
# Uses job_descriptions_input.xlsx by default
python resume_matching_engine.py

# Or pass a different Excel file
python resume_matching_engine.py my_new_jds.xlsx
```

### Step 4 — Read the output

```
JD-4 — Google (Bangalore) (Data Engineer)
Karan Mehta(0.61), Arjun Sharma(0.55), Ananya Krishnan(0.42)
```

Each line shows the Top 3 candidates ranked by cosine similarity score (0.00 to 1.00). Higher = better match.

---

## Skill Reference — Canonical Names

Use these exact names in the Excel file. Common ones grouped by category:

### Languages

| What you type  | Canonical name |
| -------------- | -------------- |
| python, pyhton | `python`     |
| java           | `java`       |
| javascript, js | `javascript` |
| typescript     | `typescript` |
| c++, cpp       | `cpp`        |
| kotlin         | `kotlin`     |
| r              | `r`          |

### ML / Data Science

| What you type                 | Canonical name          |
| ----------------------------- | ----------------------- |
| machine learning, ml, sklearn | `machine_learning`    |
| deep learning, deep-learning  | `deep_learning`       |
| tensorflow                    | `tensorflow`          |
| pytorch                       | `pytorch`             |
| keras                         | `keras`               |
| nlp                           | `nlp`                 |
| bert                          | `bert`                |
| xgboost                       | `xgboost`             |
| feature engineering           | `feature_engineering` |
| statistics                    | `statistics`          |
| regression                    | `regression`          |
| clustering                    | `clustering`          |
| matplotlib, tableau, power-bi | `data_visualization`  |
| pandas                        | `pandas`              |
| numpy                         | `numpy`               |

### Web — Frontend

| What you type       | Canonical name |
| ------------------- | -------------- |
| react, reactjs      | `react`      |
| vue, vue.js         | `vue`        |
| redux               | `redux`      |
| tailwind            | `tailwind`   |
| html/css, html, css | `html_css`   |
| jest                | `jest`       |
| graphql             | `graphql`    |

### Web — Backend

| What you type           | Canonical name    |
| ----------------------- | ----------------- |
| node.js, nodejs         | `nodejs`        |
| flask                   | `flask`         |
| spring boot, springboot | `spring_boot`   |
| rest api, rest          | `rest_api`      |
| microservices           | `microservices` |

### Databases

| What you type        | Canonical name |
| -------------------- | -------------- |
| sql                  | `sql`        |
| mysql                | `mysql`      |
| postgresql, postgres | `postgresql` |
| mongodb              | `mongodb`    |
| redis                | `redis`      |

### DevOps / Cloud

| What you type          | Canonical name |
| ---------------------- | -------------- |
| docker                 | `docker`     |
| kubernetes, kubernates | `kubernetes` |
| ci/cd, cicd            | `ci_cd`      |
| aws                    | `aws`        |

### Mobile & Design

| What you type | Canonical name |
| ------------- | -------------- |
| android       | `android`    |
| kotlin        | `kotlin`     |
| firebase      | `firebase`   |
| figma         | `figma`      |
| ui/ux         | `ui_ux`      |

### CS Fundamentals

| What you type                   | Canonical name              |
| ------------------------------- | --------------------------- |
| algorithms, algoritms           | `algorithms`              |
| data structure, data structures | `data_structures`         |
| competitive programming         | `competitive_programming` |

---

## Understanding the Score

| Score Range  | What it means                                                  |
| ------------ | -------------------------------------------------------------- |
| 0.70 – 1.00 | Strong match — candidate has most required + preferred skills |
| 0.40 – 0.69 | Decent match — covers core skills, some gaps                  |
| 0.10 – 0.39 | Weak match — partial overlap only                             |
| 0.00 – 0.09 | Poor match — very few relevant skills                         |

Scores are cosine similarity values between the resume's TF-IDF vector and the JD's binary skill vector. A score of 1.0 would mean a perfect overlap.

---

## How the Matching Works (Quick Summary)

```
Raw Skills  →  Normalize  →  Deduplicate  →  TF-IDF Vector  ──┐
                                                               ├──→  Cosine Similarity  →  Rank Top 3
JD Skills   →  Canonical names  →  Binary Vector  ────────────┘
```

1. **Normalize** — typos and alternate spellings are mapped to canonical skill names
2. **TF** = `1 / N` where N = number of unique skills in the resume
3. **IDF** = `ln(10 / df)` where df = how many resumes contain that skill
4. **JD Vector** — binary (1 if skill required/preferred, 0 otherwise)
5. **Cosine similarity** — dot product divided by product of magnitudes

---

## Common Mistakes to Avoid

* **Don't use raw/noisy names in the Excel file** — the JD parser does NOT alias-map JD skills, only resume skills are normalized. Always use canonical names in the Excel file.
* **Don't leave the Required Skills column empty** — the JD will score 0 against everything.
* **Skills are case-sensitive in the Excel** — use lowercase canonical names exactly as shown in the table above.
* **Don't add new skills not in the alias map** — they won't appear in the vocabulary and will be ignored.
