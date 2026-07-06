#!/usr/bin/env bash
set -euo pipefail

# Build the revised 24-row obfuscation stress-test dataset.
# Default: 6 examples per malicious stratum, including one Base64-encoded
# payload per stratum.
python scripts/build_obfuscated_injection_stress_test.py

# Run all obfuscation-only conditions.
python experiments_obfuscation/run_c1_obfuscation.py
python experiments_obfuscation/run_c2_obfuscation.py
python experiments_obfuscation/run_c3_obfuscation.py
python experiments_obfuscation/run_c5a_obfuscation.py
python experiments_obfuscation/run_c5b_obfuscation.py
python experiments_obfuscation/run_c5c_obfuscation.py
