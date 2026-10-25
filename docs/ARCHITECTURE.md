# Architecture

## 1. Problem formulation

Let $\mathcal{P}$ be the set of all US graduate programs, and $K_s \subset \mathcal{P}$ the programs student $s$ already knows ("anchors"). Each program $p$ has

- an attribute vector $x_p \in [0,1]^d$ on $d$ **need dimensions** (`src/aisc/needs.py`);
- a **label** $\ell_p$: its 6-digit NCES CIP 2020 code.

Each student has a latent need vector $w_s \in [0,1]^d$ with uncertainty $\sigma_s$.

**Goal.** Find programs in $\mathcal{P} \setminus K_s$ that score high on fit *and* that the student would not have found unaided.

The key design choice is to separate two spaces:

| Space | Represents | Used for |
|---|---|---|
| Need space ($x_p$, $w_s$) | what the program teaches and leads to | **relevance** |
| Label space ($\ell_p$) | what the program is *called* / where it sits in the taxonomy | **novelty** |

A good discovery is close in need space and far in label space from $K_s$.

## 2. Layers

### L0 — Program knowledge base
- Backbone: NCES CIP 2020 taxonomy; IPEDS Completions lists which institutions award master's degrees (award level 7) under each 6-digit CIP.
- Attributes $x_p$: extracted from program websites (curriculum, career outcomes) — planned LLM-assisted extraction with human spot checks.
- Logistics fields used as **hard constraints**, not dimensions: `spring_intake`, `stem_opt`, `tuition_usd`, `duration_months`.

### L1 — Student profiling (`profiling.py`)
- **Anchor decomposition** (implemented): $\hat{w}^{anchor} = \frac{1}{|K_s|}\sum_{p \in K_s} x_p$, blended with stated needs by precision weighting:
  $$\hat{w} = \frac{w^{stated}/\sigma_{stated}^2 + \hat{w}^{anchor}/\sigma_{anchor}^2}{1/\sigma_{stated}^2 + 1/\sigma_{anchor}^2}, \quad \sigma_{anchor} = \sigma_0/\sqrt{|K_s|}$$
- **Adaptive questionnaire** (M2): choose the next question to maximally reduce posterior uncertainty on the dimensions that most change the ranking.

### L2 — Discovery (`discovery.py`)
$$\text{rel}(p) = \cos(\hat{w}, x_p), \qquad \text{nov}(p) = 1 - \max_{a \in K_s} \text{sim}_\ell(p, a)$$
with $\text{sim}_\ell = 1$ for the same 4-digit CIP group, $0.5$ for the same 2-digit family, else $0$.
$$\text{score}(p) = (1-\alpha)\,\text{rel}(p) + \alpha\,\text{nov}(p)$$
Final list by Maximal Marginal Relevance: at each step pick
$\arg\max_p \,(1-\lambda)\,\text{score}(p) - \lambda \max_{q \in \text{selected}} \cos(x_p, x_q)$.

### L3 — Admission feasibility (M4)
$p_j = P(\text{admit} \mid s, j)$. Data is sparse and self-reported (GradCafe), so the plan is a Bayesian model with program-level priors, reporting intervals rather than point estimates.

### L4 — Application portfolio (`portfolio.py`)
Kapur & Hong (2022). Maximise $v(S) = \mathbb{E}[\max_{j \in S,\ \text{admitted}} u_j]$. With $S$ sorted by descending utility:
$$v(S) = \sum_i u_{j_i}\, p_{j_i} \prod_{l<i} (1-p_{j_l})$$
- Identical costs, at most $h$ applications: greedy is optimal, solutions nested in $h$.
- Integer costs $c_j$, budget $H$: exact DP over schools sorted by ascending utility,
  $V_j(b) = \max\{V_{j-1}(b),\ p_j u_j + (1-p_j)V_{j-1}(b-c_j)\}$, $O(mH)$.

## 3. Module map

| Module | Layer | Status |
|---|---|---|
| `needs.py` | shared | v0 dimensions |
| `schema.py` | L0/L1 | done |
| `profiling.py` | L1 | anchor decomposition |
| `discovery.py` | L2 | done |
| `portfolio.py` | L4 | done, verified |
