# Generic RAFM count-parity background model

## Problem
The automatic flux-wire count path is improving, but the generic RAFM sample path still has large raw `GROSS/NET` mismatches in crowded low-energy regions, especially `Ta182` and some `W187` lines.

## Current diagnosis
Two separate behaviors are still present:
- low-energy crowded `Ta182` lines can trigger overly broad comparison windows
- some `W187` lines have acceptable support windows but the local background estimate is still too low or too high, which distorts net counts

## Implemented changes
- generic RAFM comparison windows now use a linear/trapezoid local background model instead of a flat local mean
- the broad-window override is now capped by comparing widened gross counts against the primary raw ROI gross counts

## Why this is still open
- the generic RAFM path has not yet been validated to the same standard as the flux-wire count path
- some `Ta182` lines still appear to need a better crowded-region support rule
- some `W187` lines still look like a continuum-estimation problem rather than a missing-peak problem

## Exit criteria
- RAFM3 and RAFM4 raw-vs-QG line diagnostics show the main matched lines within the configured count thresholds
- the per-sample reports stop flagging the large low-energy `Ta182` and `W187` count-parity failures
- only after that should the activity/efficiency audit be treated as the next blocker
