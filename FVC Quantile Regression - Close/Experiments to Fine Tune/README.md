# Diagnostic Summary — "Floating Curve" Syndrome

- Metrics: R² ≈ 0.91 (excellent ranking), RMSE ≈ 245 (unacceptably high)
- Symptom: model captures relative decline (shape) but is biased by an additive offset — the predicted curve is “floating” above/below the true baseline (intercept error).

---

## 1. Diagnosis
- High R²: model correctly orders patients and learns decline dynamics (U‑Net features working).
- High RMSE: model has per‑patient bias at Week 0 (Baseline) that persists across time — it learns slope but not the anchor/intercept.

Root cause: Baseline FVC was provided as a regular input feature. Neural nets approximate rather than exactly reproduce the anchor, introducing a patient‑specific additive error.

---

## 2. High‑Level Fix Strategies (no full retrain required)

### Strategy A — Post‑processing Anchoring (fastest, zero retrain)
Logic:
- At Week 0 the true error must be zero. Estimate the per‑patient shift at baseline and apply it to all future predictions for that patient.

Algorithm (OOF predictions):
1. For each patient_id:
    - Find true_baseline = y_true where week==0
    - pred_baseline = y_pred where week==0
    - shift = true_baseline - pred_baseline
    - For all rows of that patient: y_pred_adj = y_pred + shift
2. Recompute RMSE/LLL on adjusted predictions.

Expected impact: commonly reduces RMSE by ~30–50 points; addresses the intercept/bias directly.

Pseudocode:
```
for pid in unique(patient_id):
     shift = y_true[(patient_id==pid) & (week==0)] - y_pred[(patient_id==pid) & (week==0)]
     y_pred[(patient_id==pid)] += shift
```

Notes:
- If Week 0 missing in some folds, use earliest available timepoint or treat separately.
- Apply the same shift to every horizon for that patient.

---

### Strategy B — Uncertainty Clipping (calibrate σ for LLL)
Problem:
- LLL penalizes being wrong and being overconfident/wrong. After anchoring, RMSE will drop but model σ may remain overly large (pessimistic).

Solution:
- After anchoring, scale predicted σ by a factor α ∈ (0.6, 1.0]; try α = 0.9, 0.8, 0.7 and pick the value that maximizes LLL on OOF holdout.
- Apply clipping or minimum σ if needed to avoid numerical issues.

Pseudocode:
```
sigma_adj = sigma_pred * alpha
```

Recommendation:
- Grid search α on OOF set and pick the best LLL. Recompute LLL after each candidate.

---

### Strategy C — Decay Factor / Ratio Target (architectural shift; retrain)
Problem persists after A/B: model still has variance because absolute FVC range is large (1000–6000 mL).

Idea:
- Predict relative change instead of absolute FVC.
- Target = FVC_current / FVC_baseline (or percent remaining). Baseline becomes 1.0 by construction; model only learns decay dynamics.

Training / inference:
- Train network to predict ratio r_t.
- At inference: FVC_pred = r_pred * real_baseline_FVC.

Benefits:
- Removes need for intercept learning, reduces variance, enforces hard anchor at inference.
- Likely produces lower RMSE and better calibrated σ when combined with Strategy B.

---

## 3. Recommended Plan (practical, low disruption)
1. Do NOT retrain yet. Export OOF predictions + σ.
2. Apply Strategy A to OOF preds. Measure RMSE and LLL.
3. If RMSE improves, tune σ scaling factor (Strategy B) on OOF to maximize LLL.
    - Try α ∈ {1.0, 0.95, 0.9, 0.85, 0.8}
4. If RMSE still above target (~170) or anchoring insufficient, implement Strategy C and retrain model on ratio target.

---

## 4. Expected Outcomes
- Anchoring: large immediate RMSE reduction (typical: −30 to −50 RMSE).
- σ scaling: improves LLL once RMSE is reduced.
- Ratio target: stable further RMSE improvement and better calibration for final model.

---

## 5. Implementation Caveats
- Ensure baseline identification is correct (week==0 or configured baseline).
- For held‑out test runs where true baseline is unknown, you can still anchor using recorded baseline in the test metadata (if available); otherwise use Strategy C at training time.
- Maintain reproducibility: version the script that applies anchoring and sigma scaling to predictions.

---

Concise summary: anchor your OOF predictions to the known baseline (zero retrain), then tighten uncertainty estimates to reflect improved error — only then consider switching the target to ratio/pct‑remaining if further gains are needed.  