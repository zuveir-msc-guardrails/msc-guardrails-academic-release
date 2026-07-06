# Time, cost, and complexity chart outputs

## Total estimated cost by condition

Compares total estimated API cost across conditions.

![Total estimated cost by condition](01_total_cost_by_condition.png)

## Mean estimated cost per row

Compares average estimated cost per example.

![Mean estimated cost per row](02_mean_cost_per_row.png)

## Mean and P95 pipeline latency

Compares average and tail latency by condition.

![Mean and P95 pipeline latency](03_mean_and_p95_latency.png)

## Total token use by condition

Compares total pipeline token use across the full run.

![Total token use by condition](04_total_tokens_by_condition.png)

## Cost breakdown by component

Shows how detector, sanitiser, and backend calls contribute to total cost.

![Cost breakdown by component](05_component_cost_breakdown.png)

## Token breakdown by component

Shows token consumption by detector, sanitiser, and backend components.

![Token breakdown by component](06_component_token_breakdown.png)

## Estimated LLM calls

Compares the number of estimated LLM calls required by each architecture.

![Estimated LLM calls](07_estimated_llm_calls.png)

## Mean architectural complexity score

Compares architectural complexity across conditions.

![Mean architectural complexity score](08_mean_complexity_score.png)

## Pipeline action counts

Shows final route counts such as allow, sanitised context, and blocked fallback.

![Pipeline action counts](09_pipeline_action_counts.png)

## Complexity path counts

Breaks each condition down into operational paths, such as classifier allow, sanitised backend, or fallback block.

![Complexity path counts](10_complexity_path_counts.png)

## ASR versus overall UA

Plots targeted malicious ASR against overall utility under attack. Best trade-off is bottom-right.

![ASR versus overall UA](11_asr_vs_overall_ua.png)

## ASR versus overall UA, zoomed

Zoomed security-utility trade-off chart to separate the high-UA, low-ASR conditions.

![ASR versus overall UA, zoomed](12_asr_vs_overall_ua_zoomed.png)

## Cost versus overall UA

Plots mean cost per row against overall utility-under-attack rate. Useful for arguing cost-effectiveness.

![Cost versus overall UA](13_cost_vs_overall_ua.png)

## Latency versus overall UA

Plots mean pipeline latency per row against overall utility-under-attack rate.

![Latency versus overall UA](14_latency_vs_overall_ua.png)

## Complexity versus overall UA

Plots descriptive architectural complexity score against overall utility-under-attack rate.

![Complexity versus overall UA](15_complexity_vs_overall_ua.png)

## LLM calls versus overall UA

Plots mean LLM calls per row against overall utility-under-attack rate.

![LLM calls versus overall UA](16_llm_calls_vs_overall_ua.png)

## Cost versus targeted ASR

Plots mean cost per row against targeted attack success rate. Lower ASR is better.

![Cost versus targeted ASR](17_cost_vs_malicious_asr.png)

## Sanitiser-stage cost

Compares additional sanitiser-stage cost for C5a, C5b, and C5c.

![Sanitiser-stage cost](18_sanitiser_cost_by_condition.png)
