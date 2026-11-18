# Completed Qwen3-4B same-worker control

Verified all 800 control answers on the same 200 sources, FP16 on L4.

| Layout | Condition | F1 (0–100) | EM (%) |
|---|---|---:|---:|
| AA | blocked_off | 43.01 | 35.50 |
| AA | full_off | 45.90 | 39.50 |
| BB | blocked_off | 42.59 | 33.50 |
| BB | full_off | 38.82 | 31.50 |
| AA+BB | blocked_off | 42.80 | 34.50 |
| AA+BB | full_off | 42.36 | 35.50 |

Paired F1 differences (percentage points); 95% unadjusted source-bootstrap intervals:

- control_AA_BB_full_minus_blocked: -0.44 [-3.62, +2.75]
- control_AA_full_minus_blocked: +2.89 [-1.39, +7.30]
- control_BB_full_minus_blocked: -3.78 [-8.75, +1.25]
- main_AB_BA_full_minus_blocked: +6.18 [+2.57, +9.85]
- split_gap_minus_together_gap: +6.62 [+1.97, +11.28]

The interaction compares split and same-worker gaps on matched sources. A near-zero control estimate is not by itself an equivalence claim. Annotation ambiguity remains; no answers were manually changed.
