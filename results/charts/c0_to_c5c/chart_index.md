# C0–C5b chart outputs, including C5a

## Security and utility by condition

Compares malicious attack success rate, malicious utility under attack, and overall task success across C0, C1, C2, C3, C5a, and C5b.

![Security and utility by condition](01_overall_security_utility.png)

## Security-utility trade-off: ASR versus UA

Plots targeted malicious ASR on the y-axis against overall UA on the x-axis for each condition.

![Security-utility trade-off: ASR versus UA](02b_asr_vs_ua_tradeoff.png)

## Targeted ASR on malicious rows

Shows the targeted attack success rate for each condition using malicious rows only as the denominator.

![Targeted ASR on malicious rows](02_malicious_asr_by_condition.png)

## Task success by condition

Compares task success on benign rows, malicious rows, and all rows.

![Task success by condition](03_task_success_breakdown.png)

## Detector precision, recall, and F1

Compares detector precision, recall, and F1 for conditions that log guardrail decisions.

![Detector precision, recall, and F1](04_detector_precision_recall_f1.png)

## Classifier block rate versus final full-block rate

Shows the difference between guardrail/classifier block decisions and actual end-to-end full blocks.

![Classifier block rate versus final full-block rate](05_classifier_vs_final_block_rate.png)

## Score matrix counts

Shows the final human-scored outcome combinations for each condition.

![Score matrix counts](06_score_matrix_counts.png)

## ASR by malicious stratum

Compares malicious-row attack success rates by attack stratum and condition.

![ASR by malicious stratum](07_asr_by_malicious_stratum.png)

## C5a, C5b, and C5c task success by stratum

Compares task success by stratum for C5a, C5b, and C5c.

![C5a, C5b, and C5c task success by stratum](08_c5a_c5b_task_success_by_stratum.png)

## C5a, C5b, and C5c sanitisation flow counts

Shows attempted, succeeded, fallback-blocked, and backend-after-sanitisation counts for C5a, C5b, and C5c.

![C5a, C5b, and C5c sanitisation flow counts](09_c5a_c5b_c5c_sanitisation_flow_counts.png)

## C5a, C5b, and C5c sanitised malicious outcomes

Shows attack success, task success, and UA among malicious rows sanitised by C5a, C5b, and C5c.

![C5a, C5b, and C5c sanitised malicious outcomes](10_c5a_c5b_c5c_sanitised_malicious_outcomes.png)

## C5a, C5b, and C5c sanitised malicious UA by stratum

Shows UA among sanitised malicious rows, broken down by attack stratum for C5a, C5b, and C5c.

![C5a, C5b, and C5c sanitised malicious UA by stratum](11_c5a_c5b_c5c_sanitised_ua_by_stratum.png)

## C5a and C5c unit-level removal burden

Shows the number of units removed and retained by C5a and C5c.

![C5a and C5c unit-level removal burden](12_c5a_c5c_unit_removal_counts.png)
