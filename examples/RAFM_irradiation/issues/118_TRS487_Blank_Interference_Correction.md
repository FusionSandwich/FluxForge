# Issue: Implementation: TRS-487 Spectrum Interferences & Blank Handling

**Status:** Planned
**Context:** Deep Research Report Standardization

## Description & Implementation Mechanics
**Target:** `src/fluxforge/qaqc/interference.py`
**Goal:** Explicit logic tagging known threshold matrix reactions (e.g. Al(n,a)Na-24 masking independent Na results) and generating Blank subtraction margins as defined uniformly by TRS 487 guidance.
