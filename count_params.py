"""Print the parameter count of DPR-Net and the baseline, by component."""
from dprnet import build, count_parameters

for name in ("dprnet", "baseline"):
    m = build(name)
    parts = {k: count_parameters(getattr(m, k)) if getattr(m, k) is not None else 0
             for k in ("enc", "pea", "context", "sfb", "dec")}
    parts["line+fusion"] = sum(count_parameters(getattr(m, k)) for k in ("line_h", "line_v", "line", "fusion"))
    parts["refine+heads"] = sum(count_parameters(getattr(m, k)) for k in ("refine", "head", "aux_head"))
    print(f"{name:9s} total {count_parameters(m):>10,}  " + "  ".join(f"{k} {v:,}" for k, v in parts.items()))
