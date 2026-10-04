"""Check the latest published comparator without changing the working source."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

out=Path(__file__).resolve().parent
repo=out.parents[2]
results=[]
for ref in ("HEAD","origin/codex/adversarial-acceptance-20261002"):
    revision=subprocess.check_output(["git","rev-parse",ref],cwd=repo,text=True).strip()
    source=subprocess.check_output(["git","show",f"{ref}:src/fluxforge/validation/reference_parity.py"],cwd=repo)
    parsed=ast.parse(source)
    module=ast.Module(body=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in {"_values_close","_resolve_tolerance"}],type_ignores=[])
    namespace={}
    exec(compile(module,"published_comparator","exec"),namespace)
    checks=[]
    for field in ("energies_keV[0]","first_peak_keV"):
        field_key=field.split("[")[0]
        for suffix,broad,explicit in (("abs",8.0,0.01),("rel",0.1,0.001)):
            values={f"energy_keV_{suffix}":broad,f"{field_key}_{suffix}":explicit}
            for keys in (list(values),list(reversed(values))):
                limits={k:values[k] for k in keys}
                matched=namespace["_values_close"](200.0,201.0,path=f"output.{field}",tolerances=limits)
                checks.append({"field":field,"suffix":suffix,"key_order":keys,"expected_match":False,"observed_match":matched,"correct":not matched})
    results.append({"ref":ref,"git_head":revision,"source_sha256":hashlib.sha256(source).hexdigest(),"checks":checks,"correct":sum(c["correct"] for c in checks)})
(out/"published_parity_precedence.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps([{k:r[k] for k in ("ref","git_head","correct")} for r in results]))
assert results[0]["correct"]==8
