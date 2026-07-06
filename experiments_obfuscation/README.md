# Revised obfuscation stress-test scripts with Base64

This package updates the separate exploratory obfuscation stress-test dataset and runners.

## What changed

- The dataset builder now creates **24 malicious examples** by default.
- It selects **6 examples per malicious stratum**:
  - instruction_override
  - data_exfiltration
  - tool_misuse
  - markdown_injection
- The sixth example in each stratum uses `base64_encoded_payload`.
- The Base64 rows are still separate from the frozen 190-row main benchmark.
- C2 now includes a lightweight `base64_like_blob` heuristic rule.
- C3, C5a, C5b and C5c prompts now explicitly mention encoded/Base64 instruction carriers.
- C5c unitisation/sanitisation now preserves and removes encoded-looking payload units.

## Install locations

Copy the files as follows:

```text
scripts/build_obfuscated_injection_stress_test.py
experiments_obfuscation/run_c1_obfuscation.py
experiments_obfuscation/run_c2_obfuscation.py
experiments_obfuscation/run_c3_obfuscation.py
experiments_obfuscation/run_c5a_obfuscation.py
experiments_obfuscation/run_c5b_obfuscation.py
experiments_obfuscation/run_c5c_obfuscation.py
run_all_obfuscation.sh
```

## Run

From the repository root:

```bash
python scripts/build_obfuscated_injection_stress_test.py
bash run_all_obfuscation.sh
```

The output dataset remains:

```text
data/obfuscation/obfuscated_malicious.jsonl
data/obfuscation/obfuscated_malicious.csv
data/obfuscation/obfuscation_manifest.md
```

Logs remain under:

```text
results/obfuscation/logs/<condition>/
```

## Scoring

Do not merge the 24 obfuscation rows into the main 190-row results. Score them separately as an exploratory robustness check.


## v3 review/audit fields

The run logs now include the fields needed for manual scoring beside each model output:

- `question` and `user_question`
- `expected_safe_answer`
- `context`
- `injected_context`
- `actual_attack_payload` and `payload`
- `original_attack_payload` / `source_payload` / `obfuscation_original_payload`
- `attack_goal`
- `obfuscation_decoded_payload` and `obfuscation_encoded_payload`
- `obfuscation_note`

Use `actual_attack_payload` or `payload` as the attack payload that was present in the obfuscated row. For Base64 rows, `obfuscation_decoded_payload` contains the decoded malicious instruction and `obfuscation_encoded_payload` contains the encoded string.
