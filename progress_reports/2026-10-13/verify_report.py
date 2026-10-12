"""Check page count and all 120 rounded LaTeX matrix entries against raw records."""
import json
import re
from pathlib import Path
from pypdf import PdfReader

root = Path(__file__).resolve().parent
tex = (root / "progress_report.tex").read_text(encoding="utf-8")
data = json.loads((root / "numerical_results.json").read_text(encoding="utf-8"))
rows = data["experiments"]["pc_routes_correctness.json"]
assert len(rows) == len({r["id"] for r in rows}) == 120
assert sum(r["test_budget_passed"] for r in rows) == 53
for metric in ("latency", "energy"):
    assert sum(r["test_budget_passed"] and r[metric + "_saving_vs_best_eligible_static_percent"] > 0 for r in rows) == 31
section = tex.split("% BEGIN_CORRECTNESS_MATRIX", 1)[1].split("% END_CORRECTNESS_MATRIX", 1)[0]
objective = None
checked = 0
for line in section.splitlines():
    parts = line.removesuffix(r"\\").split("&")
    if len(parts) != 6 or not parts[1].isdigit():
        continue
    if parts[0]:
        objective = parts[0].lower()
    batch, budget = int(parts[1]), float(parts[2])
    for seed, cell in zip((42, 45, 48), parts[3:]):
        record = next(r for r in rows if (r["objective"], r["batch_size"], r["budget_pp"], r["seed"]) == (objective, batch, budget, seed))
        values = [float(x) for x in re.findall(r"[+-]?\d+\.\d+", cell)]
        assert len(values) == 3, cell
        expected = [record["accuracy_loss_vs_p0_pp"], record["latency_saving_vs_best_eligible_static_percent"], record["energy_saving_vs_best_eligible_static_percent"]]
        assert all(abs(a - b) <= tol for a, b, tol in zip(values, expected, (0.00501, 0.05001, 0.05001))), (record["id"], values, expected)
        assert ("$^*$" in cell) == (not record["test_budget_passed"]), record["id"]
        checked += 1
assert checked == 120, checked
pdf = PdfReader(root / "progress_report.pdf")
assert len(pdf.pages) == 5, len(pdf.pages)
assert all(str(i) in page.extract_text() for i, page in enumerate(pdf.pages, 1))
assert "References" in pdf.pages[-1].extract_text()
print("PASS: five PDF pages; all 120 matrix cells, failure markers, 53 passes and 31/31 positive counts match saved records.")
