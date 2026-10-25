# Related work

## Application portfolio optimisation
| Work | What it does | How we use it |
|---|---|---|
| Kapur & Hong (2022), *The College Application Problem*, arXiv:2205.01869 | Formulates application as maximising expected max utility under a budget; greedy optimal for identical costs, NP-complete otherwise; DP / FPTAS / simulated annealing | Basis of `portfolio.py`; test A2 reproduces their README example |
| [maxkapur/OptimalApplication.jl](https://github.com/maxkapur/OptimalApplication.jl) | Julia implementation of the above | Reference results |
| [maxkapur/CollegeApplication](https://github.com/maxkapur/CollegeApplication) | MS thesis repo (Seoul National University) | Proofs and complexity results |

## Admission prediction
| Work | Notes |
|---|---|
| Many GitHub repos on the Kaggle "Graduate Admissions" dataset (Acharya, 2018) | Regression on GRE / TOEFL / CGPA. Target is a synthetic "chance of admit", so not usable for calibration. |
| Heidari Far & Tabrizi (2026), arXiv:2603.29881 | ~13,000 GradCafe records (2021–2025) enriched with OpenAlex, QS subject rankings, Wikidata; admission prediction + university-program recommendation |
| Zhao et al. (2023), PLOS ONE 10.1371/journal.pone.0291107 | ML on graduate admissions including recommendation letters; SHAP feature importance |

## Program recommendation / decision support
| Work | Notes |
|---|---|
| arXiv:2501.06581 | Study-program recommender using program content (topic modelling) + personal preferences on public data |
| AHP / TOPSIS university selection systems (e.g. Wongvilaisakul et al., 2023, *Sustainability*) | Multi-criteria weighting; useful for eliciting utility weights |

## Conversational preference elicitation (cold start)
| Work | Notes |
|---|---|
| Christakopoulou, Radlinski & Hofmann (2016), *Towards Conversational Recommender Systems*, KDD | Choosing which questions to ask a new user; ~25% improvement after 2 questions; bandit explore–exploit |
| Zuo et al. (2022), *Hierarchical Conversational Preference Elicitation with Bandit Feedback*, CIKM | Ask about categories (key-terms) or recommend items; Hier-UCB / Hier-LinUCB. Maps to "ask about needs before naming programs" |

## Data sources
| Source | Use | Caveat |
|---|---|---|
| NCES CIP 2020 | Program taxonomy (label space) | Coarse: one code covers very different programs |
| IPEDS Completions | Which institutions award master's (award level 7) per 6-digit CIP | Counts only, no curriculum |
| Program websites | Attributes | Must record URL + retrieval date |
| GradCafe | Admission outcomes | Self-reported, selection bias; check terms of service before any collection |
