"""Compare two offline RAFM software receipts without loosening physics equality."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logical_path(path):
    normalized = path.replace("\\", "/")
    for prefix in ["/examples/RAFM_irradiation/", "/docs/reviews/"]:
        if prefix in normalized:
            return normalized[normalized.index(prefix) + 1 :]
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before, after = [json.loads(p.read_text()) for p in [args.before, args.after]]
    assert before["validation_script_sha256"] == after["validation_script_sha256"]
    inputs = [
        {logical_path(p): (p, digest) for p, digest in r["input_sha256"].items()}
        for r in [before, after]
    ]
    assert inputs[0].keys() == inputs[1].keys()
    document_endings = []
    for key in inputs[0]:
        (p1, h1), (p2, h2) = inputs[0][key], inputs[1][key]
        assert sha(p1) == h1 and sha(p2) == h2
        if h1 != h2:
            assert key.endswith(".md")
            assert Path(p1).read_bytes().replace(b"\r\n", b"\n") == Path(
                p2
            ).read_bytes().replace(b"\r\n", b"\n")
            document_endings.append(key)
    exact = [
        "sources",
        "edges_eV",
        "row_cross_sections_barn",
        "row_uncertainties_barn",
        "group_integral_prior",
        "forward_predictions",
        "units",
        "assumptions",
        "limitations",
    ]
    assert all(before[k] == after[k] for k in exact)
    workflows = {}
    for key in before["workflows"]:
        a, b = before["workflows"][key], after["workflows"][key]
        for field in ["response", "rates", "flux", "predictions"]:
            np.testing.assert_allclose(a[field], b[field], rtol=1e-12, atol=0.0)
        delta = np.abs(np.array(a["predictions"]) / np.array(b["predictions"]) - 1.0)
        workflows[key] = dict(
            response_bitwise_equal=a["response"] == b["response"],
            max_prediction_relative_delta=float(delta.max()),
        )
    assert all(v["output_rows"] == 2 and not v["equal_keys"] for v in after["variants"])
    assert all(v["rejected"] for v in after["invalid_copies"])
    result = dict(
        before_sha256=sha(args.before),
        after_sha256=sha(args.after),
        before_commit=before["code_commit"],
        after_commit=after["code_commit"],
        exact_fields=exact,
        identical_data_hashes=True,
        reference_document_line_ending_differences=document_endings,
        workflow_comparison_rtol=1e-12,
        workflow_comparison_atol=0.0,
        workflows=workflows,
        before_variant_rows=[v["output_rows"] for v in before["variants"]],
        after_variant_rows=[v["output_rows"] for v in after["variants"]],
        after_invalid_rejected=sum(v["rejected"] for v in after["invalid_copies"]),
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
