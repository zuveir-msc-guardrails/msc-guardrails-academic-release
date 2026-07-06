
import json

# Load a real BIPIA email context
with open('external/BIPIA/benchmark/email/test.jsonl') as f:
    email_contexts = [json.loads(line) for line in f if line.strip()]

# Load a real BIPIA text attack payload
with open('external/BIPIA/benchmark/text_attack_test.json') as f:
    attacks = json.load(f)

# Print email context structure
print("EMAIL CONTEXT FIELDS:", list(email_contexts[0].keys()))
print(email_contexts[0])

# Print attack payload structure
print("\nATTACK TYPE (top-level keys):", list(attacks.keys()))

# Get first attack category and first payload
first_category = list(attacks.keys())[0]
print(f"\nFirst category: {first_category}")
print(f"Number of attacks in this category: {len(attacks[first_category])}")
print(f"\nFirst attack payload: {attacks[first_category][0]}")

# Print all categories
print("\nAll attack categories:", list(attacks.keys()))

# Build one combined example — naive injection style
context = email_contexts[0]['context']
question = email_contexts[0]['question']
ideal    = email_contexts[0]['ideal']

# Pick a clear instruction override payload
payload = attacks['Language Translation'][0]
print(f"\nPayload: {payload}")

# Naive injection — append directly
injected = context + "\n\n" + payload
print(f"\nInjected context:\n{injected}")
print(f"\nUser question: {question}")
print(f"\nExpected safe answer: {ideal}")