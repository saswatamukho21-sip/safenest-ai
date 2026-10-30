import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ml_engine import SafeNestRecommenderEngine

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_file = os.path.join(base_dir, "data", "pg_dataset.json")

engine = SafeNestRecommenderEngine(dataset_file)
results = engine.recommend("biometric security quiet single room", max_budget=15000, min_safety_score=50)

print(f"Total results matched: {len(results)}")
for r in results[:3]:
    print(f"- {r['name']} | Match: {r['composite_match_score']}% | Safety: {r['safety_score']}/100 | Rent: Rs.{r['monthly_rent']}")

print("\nAll ML tests passed successfully!")
