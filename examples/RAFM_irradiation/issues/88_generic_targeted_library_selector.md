## Title
Use exploratory-support-driven targeted libraries for generic RAFM recovery

## Problem
The generic RAFM targeted pass was fitting against a dense all-isotope line library. In practice that let weak nuisance lines from dense isotopes, especially `Tb154m`, suppress stronger physically relevant lines such as:
- `Fe59 @ 1099.25 keV`
- `Co60 @ 1173.17 keV`

At the same time, simply restricting the library to already-identified isotopes dropped plausible but weaker products such as `W187`.

## Implemented direction
- keep full line sets for isotopes already supported by the exploratory pass
- keep strongest fallback lines for unsupported isotopes
- then prune very weak nearby nuisance lines

## Why
This is automatic, QG-independent at runtime, and more defensible than hard-disabling one isotope.

## Current status
Focused RAFM regressions now recover the previous RAFM4 missing-line set and no longer miss RAFM3-B `W187` targeted peaks.

## Remaining work
- crowded low-energy `Ta182` count parity
- RAFM3 `W187` local background modeling
- uncertainty audit after count parity stabilizes
