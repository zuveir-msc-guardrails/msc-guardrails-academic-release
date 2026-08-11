#!/usr/bin/env bash
set -euo pipefail

# Run the supplementary obfuscation stress test using the supplied
# 24-row dataset in data/obfuscation/obfuscated_malicious.jsonl.
#
# The dataset-construction script is retained separately for provenance
# and regeneration, but is not run automatically here so that the
# supplied evaluation dataset remains unchanged during reproduction.
# python scripts/build_obfuscated_injection_stress_test.py

# Run all obfuscation-only conditions.
python experiments_obfuscation/run_c1_obfuscation.py
python experiments_obfuscation/run_c2_obfuscation.py
python experiments_obfuscation/run_c3_obfuscation.py
python experiments_obfuscation/run_c5a_obfuscation.py
python experiments_obfuscation/run_c5b_obfuscation.py
python experiments_obfuscation/run_c5c_obfuscation.py
