# Reviewed example engine epochs

`qg_sensitivity_review.json` binds the six reused engine files to immutable Git revision `be4e7f578803265ed0df0063d634086f16c7d7da` using canonical LF hashes. The QG protocol example and Co-Cd joint-Poisson pilot default to this integrated profile. Their new outputs record this binding, the observed checkout identity, and their original historical pins separately.

Use `--engine-profile historical` to enforce the original a7bcc68 source hashes. It intentionally rejects a different current engine. The original fixture manifest and historical reports remain unchanged.

In Git checkouts, the integrated binding verifies current file hashes, ancestry, and exact blobs at the declared source revision. In portable source bundles it verifies all supplied pins and explicitly reports Git revision and ancestry as unknown. This is a reproducibility check, not scientific qualification or proof of vendor parity. The Co-Cd pilot still flags strong lack of fit and unidentifiable normalization controls.
