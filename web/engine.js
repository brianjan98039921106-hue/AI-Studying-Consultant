/* Student self-service engine (W-53, D14): the arithmetic of the Python pipeline,
 * re-implemented for the browser. All data come from data.json (built by
 * scripts/build_web_data.py); tests/test_web_parity.py checks that this file gives the
 * same profile, shortlist and recommendations as aisc (A-C2-60).
 *
 * Mirrors:
 *   scoreAnswers   aisc.profiling.questionnaire.Questionnaire.needs / group_scores
 *   areaOrder      aisc.profiling.hierarchy.areas_to_ask(group_scores, m=12)
 *   complete       aisc.profiling.anchors.complete_neutral (no anchors, area_model hierarchy)
 *   termsFor       aisc.discovery.scorers.DEFAULT_SCORER_V2.terms
 *   shortlist      aisc.discovery.sectioned.shortlist (no constraints, no diversity)
 *   discover       aisc.discovery.sectioned.discover after profiling.apply_familiarity
 */
(function (root) {
  "use strict";

  function makeEngine(data) {
    const keys = data.dims.map((d) => d.key);
    const layerIdx = {};
    data.scorer.layers.forEach((layer) => {
      layerIdx[layer] = [];
    });
    data.dims.forEach((d, i) => layerIdx[d.layer].push(i));
    const areaKeys = data.dims.filter((d) => d.layer === "domain").map((d) => d.key);
    const groupIds = Object.keys(data.groups).map(Number).sort((a, b) => a - b);

    function scoreAnswers(answers) {
      const acc = {};
      const groupScores = {};
      for (const it of data.items) {
        const v = answers[it.id];
        if (v === null || v === undefined) continue;
        if (!Number.isInteger(v) || v < 1 || v > 5) throw new Error(`${it.id}: answer must be 1-5`);
        const unit = (v - 1) / 4;
        if (it.stage === "group") groupScores[it.group] = unit;
        if (it.dimension) (acc[it.dimension] = acc[it.dimension] || []).push(unit);
      }
      const needs = {};
      for (const k of Object.keys(acc)) {
        needs[k] = acc[k].reduce((s, x) => s + x, 0) / acc[k].length;
      }
      return { needs, groupScores };
    }

    function areasInGroup(g) {
      return areaKeys.filter((k) => data.area_groups[k].includes(g));
    }

    function areaOrder(groupScores) {
      const groups = groupIds.slice().sort((a, b) => {
        const sa = a in groupScores ? groupScores[a] : -1;
        const sb = b in groupScores ? groupScores[b] : -1;
        return sb - sa || a - b;
      });
      const order = [];
      for (const g of groups) {
        for (const k of areasInGroup(g)) if (!order.includes(k)) order.push(k);
      }
      for (const k of areaKeys) if (!order.includes(k)) order.push(k);
      return order;
    }

    function areaPrior(key, groupScores) {
      const rated = data.area_groups[key].filter((g) => g in groupScores).map((g) => groupScores[g]);
      return rated.length ? Math.max(...rated) : data.base_rate[key];
    }

    function complete(needs, groupScores) {
      const filled = {};
      const source = {};
      for (const k of keys) {
        if (k in needs) {
          filled[k] = needs[k];
          source[k] = "answered";
        } else if (k in data.area_groups) {
          filled[k] = areaPrior(k, groupScores);
          source[k] = "prior";
        } else {
          filled[k] = data.neutral;
          source[k] = "prior";
        }
      }
      return { vector: keys.map((k) => filled[k]), source };
    }

    function termsFor(w, x) {
      const out = new Array(w.length).fill(0);
      const cw = data.scorer.cos_weight;
      const sw = data.scorer.sq_weight;
      for (const layer of data.scorer.layers) {
        const idx = layerIdx[layer];
        const weight = data.scorer.weights[layer];
        const D = idx.length;
        let nw = 0;
        let nx = 0;
        for (const i of idx) {
          nw += w[i] * w[i];
          nx += x[i] * x[i];
        }
        const norm = Math.sqrt(nx) * Math.sqrt(nw);
        for (const i of idx) {
          const cos = norm === 0 ? 0 : (x[i] * w[i]) / norm;
          const sq = (x[i] - w[i]) ** 2;
          out[i] = weight * (cw * cos + sw * (1 / D - sq / D));
        }
      }
      return out;
    }

    function score(w, x) {
      return termsFor(w, x).reduce((s, t) => s + t, 0);
    }

    function ranked(w) {
      return data.catalog
        .map((p) => ({ p, s: score(w, p.vector) }))
        .sort((a, b) => b.s - a.s || (a.p.id < b.p.id ? -1 : a.p.id > b.p.id ? 1 : 0));
    }

    function shortlist(w, n = data.flow.n_shortlist) {
      return ranked(w).slice(0, n).map((r) => r.p.id);
    }

    function discover(w, familiarity, kNew = data.flow.k_new, kKnown = data.flow.k_known) {
      for (const lvl of Object.values(familiarity)) {
        if (!data.flow.levels.includes(lvl)) throw new Error(`unknown familiarity level ${lvl}`);
      }
      const known = new Set(
        Object.keys(familiarity).filter((id) => data.flow.known.includes(familiarity[id]))
      );
      const all = ranked(w);
      const rec = (r) => ({ id: r.p.id, program: r.p, relevance: r.s, terms: termsFor(w, r.p.vector) });
      return {
        discoveries: all.filter((r) => !known.has(r.p.id)).slice(0, kNew).map(rec),
        knownGood: all.filter((r) => known.has(r.p.id)).slice(0, kKnown).map(rec),
      };
    }

    function reasons(rec, n = 3) {
      const skip = new Set(rec.program.unmeasured);
      return rec.terms
        .map((t, i) => ({ key: keys[i], label: data.dims[i].label, value: t }))
        .filter((r) => r.value > 0 && !skip.has(r.key))
        .sort((a, b) => b.value - a.value)
        .slice(0, n);
    }

    return { keys, scoreAnswers, areaOrder, complete, termsFor, score, shortlist, discover, reasons };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { makeEngine };
  else root.AiscEngine = { makeEngine };
})(typeof self !== "undefined" ? self : this);
