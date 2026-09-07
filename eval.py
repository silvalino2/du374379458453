import csv
import requests

def load_test_cases(csv_path: str, per_topic: int = 1):
    seen_topics = set()
    cases = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # ADD THIS — skips the header row
        for row in reader:
            topic, question, answer, source_url, date = row
            if topic not in seen_topics:
                cases.append({"question": question, "expected_source": source_url, "topic": topic})
                seen_topics.add(topic)
    return cases

def run_eval(csv_path: str):
    test_cases = load_test_cases(csv_path)
    correct = 0
    print(f"Running {len(test_cases)} test cases (one per topic)...\n")

    for case in test_cases:
        r = requests.get("http://localhost:8000/generate", params={"prompt": case["question"]})
        data = r.json()
        sources = [s["source"] for s in data.get("sources", [])]
        passed = case["expected_source"] in sources
        correct += passed
        print(f"{'PASS' if passed else 'FAIL'} — [{case['topic']}] {case['question']}")

    total = len(test_cases)
    print(f"\n{correct}/{total} correct ({correct/total*100:.1f}%)")

run_eval('corpus.csv')  # replace with your real filename if different