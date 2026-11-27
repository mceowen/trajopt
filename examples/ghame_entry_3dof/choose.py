"""Select the reported configuration of each transcription from the tuning screen."""
import json, os
rows = []
for m in ("MS-term", "p-term", "hp-term"):
    import glob
    for f in sorted(glob.glob(f"plots/matrix_{m}*/screen.json")): rows += json.load(open(f))
sel = {}
for m in ("MS-term", "p-term", "hp-term"):
    rs = [r for r in rows if r["method"] == m and r.get("R") is not None and (m == "MS-term" or r["control"] == "foh")]
    passing = [r for r in rs if r["pass"]]
    if passing:
        best = max(passing, key=lambda r: r["cost"])
    else:
        conv = [r for r in rs if r["converged"]] or rs
        rmin = min(r["R"] for r in conv)
        best = min([r for r in conv if r["R"] <= 1.01 * rmin], key=lambda r: r["iterations"])
    sel[m] = dict(best, screened=len(rs), passing=len(passing), converged=len([r for r in rs if r["converged"]]))
    print(f"SELECT {m}: {best['tag']} pass={best['pass']} cost={best['cost']:.3f} R={best['R']:.3g} limiting={best['limiting']} ({len(passing)} passing of {sel[m]['screened']} screened)")
os.makedirs("plots/matrix", exist_ok=True); json.dump(sel, open("plots/matrix/selected.json", "w"), indent=1, default=str)
