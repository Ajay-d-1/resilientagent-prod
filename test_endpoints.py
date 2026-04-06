#!/usr/bin/env python3
"""Quick test of all endpoints — simulates what the hackathon validator does."""
import requests
import json
import sys

BASE = "http://localhost:7860"

def test(name, method, path, json_body=None, expected_keys=None):
    """Test an endpoint and print pass/fail."""
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            if json_body is not None:
                r = requests.post(url, json=json_body, timeout=10)
            else:
                # CRITICAL: validator sends POST with NO body
                r = requests.post(url, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            if expected_keys:
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    print(f"  FAIL {name}: missing keys {missing}")
                    print(f"       Got: {json.dumps(data, indent=2)[:300]}")
                    return False
            print(f"  PASS {name} (200) -> {json.dumps(data)[:200]}")
            return True
        else:
            print(f"  FAIL {name}: HTTP {r.status_code}")
            print(f"       Body: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return False

print("=" * 60)
print("HACKATHON SUBMISSION VALIDATOR SIMULATION")
print("=" * 60)

results = []

# 1. Health check
print("\n--- Health Check ---")
results.append(test("GET /health", "GET", "/health"))

# 2. CRITICAL: Reset with NO body (exactly what the validator does)
print("\n--- Reset (empty body - validator style) ---")
results.append(test("POST /reset (no body)", "POST", "/reset", expected_keys=["observation"]))

# 3. Reset with task_id body
print("\n--- Reset (with task_id) ---")
results.append(test("POST /reset (task_id)", "POST", "/reset", 
                     json_body={"task_id": "task1_latency_spike"}, 
                     expected_keys=["observation"]))

# 4. Step
print("\n--- Step ---")
results.append(test("POST /step", "POST", "/step",
                     json_body={"action_type": "check_metrics", "target": "inference_service"},
                     expected_keys=["observation", "reward", "done"]))

# 5. State
print("\n--- State ---")
results.append(test("GET /state", "GET", "/state", expected_keys=["state"]))

# 6. Grader
print("\n--- Grader ---")
results.append(test("POST /grader", "POST", "/grader", expected_keys=["score"]))

# 7. Tasks
print("\n--- Tasks ---")
results.append(test("GET /tasks", "GET", "/tasks", expected_keys=["tasks"]))

# 8. Run a full task cycle: reset -> step sequence -> grader
print("\n--- Full Task Cycle (task1_latency_spike) ---")
r = requests.post(f"{BASE}/reset", json={"task_id": "task1_latency_spike"})
if r.status_code == 200:
    actions = [
        ("check_metrics", "inference_service"),
        ("read_logs", "inference_service"),
        ("optimize_batch", "inference_service"),
        ("verify_fix", "inference_service"),
    ]
    for act, tgt in actions:
        r = requests.post(f"{BASE}/step", json={"action_type": act, "target": tgt})
        data = r.json()
        reward = data.get("reward", "?")
        done = data.get("done", "?")
        print(f"  Step: {act}({tgt}) -> reward={reward}, done={done}")
        if done:
            break
    
    r = requests.post(f"{BASE}/grader")
    score = r.json().get("score", "?")
    print(f"  Final grade: {score}")
    results.append(score > 0.5 if isinstance(score, (int, float)) else False)

# 9. Test task2 and task3 cycles
for task_id, actions_seq in [
    ("task2_prediction_drift", [
        ("analyze_drift", "ml_model"),
        ("check_deployment", "ml_model"),
        ("rollback_model", "ml_model"),
        ("verify_fix", "ml_model"),
    ]),
    ("task3_cascading_failure", [
        ("check_metrics", "primary_model"),
        ("read_logs", "primary_model"),
        ("restart_service", "primary_model"),
        ("scale_service", "fallback_model"),
        ("verify_fix", "primary_model"),
    ]),
]:
    print(f"\n--- Full Task Cycle ({task_id}) ---")
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id})
    if r.status_code == 200:
        for act, tgt in actions_seq:
            r = requests.post(f"{BASE}/step", json={"action_type": act, "target": tgt})
            data = r.json()
            reward = data.get("reward", "?")
            done = data.get("done", "?")
            print(f"  Step: {act}({tgt}) -> reward={reward}, done={done}")
            if done:
                break
        
        r = requests.post(f"{BASE}/grader")
        score = r.json().get("score", "?")
        print(f"  Final grade: {score}")
        results.append(score > 0.5 if isinstance(score, (int, float)) else False)

print("\n" + "=" * 60)
passed = sum(1 for r in results if r)
total = len(results)
print(f"RESULTS: {passed}/{total} passed")
if passed == total:
    print("✅ ALL CHECKS PASSED — Ready for submission!")
else:
    print("❌ SOME CHECKS FAILED — Fix issues above")
print("=" * 60)
sys.exit(0 if passed == total else 1)
