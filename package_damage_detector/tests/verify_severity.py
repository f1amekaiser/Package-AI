
import sys
import os
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.decision_engine import compute_severity

@dataclass
class MockDetection:
    classifier_label: str
    classifier_confidence: float

def test_case(name, detections, expected_label, expected_score_range):
    print(f"Testing {name}...")
    result = compute_severity(detections)
    score = result["severity_score"]
    label = result["severity_label"]
    
    print(f"  Result: Score={score}, Label={label}")
    
    label_match = label == expected_label
    score_match = expected_score_range[0] <= score <= expected_score_range[1]
    
    if label_match and score_match:
        print("  ✅ PASS")
    else:
        print(f"  ❌ FAIL (Expected Label={expected_label}, Score in {expected_score_range})")

# Case 1: All intact (High confidence)
det1 = [
    MockDetection("intact", 1.0),
    MockDetection("intact", 0.83)
]
test_case("Case 1: All Intact", det1, "SAFE", (0, 5))

# Case 2: Confirmed Damage (High Severity)
det2 = [
    MockDetection("damaged", 0.90)
]
# Score = 80 + (0.9 * 20) = 98
test_case("Case 2: Severe Damage", det2, "HIGH", (98, 98))

# Case 3: Confirmed Damage (Medium Severity)
det3 = [
    MockDetection("damaged", 0.60)
]
# Score = 40 + (0.6 * 40) = 40 + 24 = 64
test_case("Case 3: Moderate Damage", det3, "MEDIUM", (64, 64))

# Case 4: Mixed (1 damage, 1 intact)
det4 = [
    MockDetection("intact", 0.99),
    MockDetection("damaged", 0.85)
]
# Max conf of *damage* is 0.85 -> 80 + 17 = 97
test_case("Case 4: Mixed", det4, "HIGH", (97, 97))

