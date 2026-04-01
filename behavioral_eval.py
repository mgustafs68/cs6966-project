import pandas as pd

# Change this to the filepath of your policy eval predictions made from `extract_policy_logits.py`
preds = pd.read_csv("~/cs6966-project/outputs/policy_eval/20260401_152337/predictions.csv")

subset = preds[preds["template_type"].isin(["AW", "NC"])]

mismatches = (subset["predicted"] != subset["gold"]).sum()
total = len(subset)

correct = (preds["predicted"] == preds["gold"]).sum()
grand_total = len(preds)

result = mismatches / total if total > 0 else 0
result2 = correct / grand_total if grand_total > 0 else 0
print("Agreement rate:", result)
print("Correction rate:", result2)