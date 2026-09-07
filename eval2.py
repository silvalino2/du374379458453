import csv
import requests

PARAPHRASES = {
    "NIN": "how do I get my NIN",
    "Passport": "how much is international passport in nigeria",
    "CAC": "do I need to register my business with CAC",
    "Tax": "is TIN still used in nigeria",
    "Driver's License": "how much to get driver's license",
    "Voter Registration": "how do I get my PVC",
    "Birth Certificate": "where do I get my child's birth certificate",
    "Tax Clearance Certificate": "what is tax clearance certificate for",
    "Land Registration": "what is C of O for land",
    "Health Insurance": "what is the new health insurance scheme in nigeria",
    "CERPAC": "do foreigners need CERPAC to stay in nigeria",
    "JAMB/WAEC": "how much is jamb form this year",
    "Pension (PenCom)": "do I need BVN for my pension account",
    "BVN": "how do I get bvn",
    "Police Character Certificate": "how do I get police clearance certificate",
    "Vehicle Registration": "how do I register my new car",
    "Marriage Registration": "how long does court marriage notice take",
}

def load_expected_sources(csv_path: str):
    seen_topics = set()
    sources = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            topic, question, answer, source_url, date = row
            if topic not in seen_topics:
                sources[topic] = source_url
                seen_topics.add(topic)
    return sources

def run_paraphrase_eval(csv_path: str):
    expected = load_expected_sources(csv_path)
    correct = 0
    total = 0
    print(f"Running {len(PARAPHRASES)} paraphrased test cases...\n")

    for topic, question in PARAPHRASES.items():
        if topic not in expected:
            print(f"SKIP — [{topic}] no source found in CSV")
            continue
        expected_source = expected[topic]
        r = requests.get("http://localhost:8000/generate", params={"prompt": question})
        data = r.json()
        sources = [s["source"] for s in data.get("sources", [])]
        passed = expected_source in sources
        correct += passed
        total += 1
        print(f"{'PASS' if passed else 'FAIL'} — [{topic}] {question}")

    print(f"\n{correct}/{total} correct ({correct/total*100:.1f}%)")

run_paraphrase_eval('corpus.csv')