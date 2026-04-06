# FluxForge Project Management Assets

This directory stores the Stage 0 tracker state as repository data so the roadmap is
reviewable and reproducible in code review.

## Files

- `milestones.json`: canonical milestone titles and descriptions.
- `labels.json`: label taxonomy for area, type, priority, platform, and special labels.
- `board.json`: project board columns and automation rules.
- `issues.json`: epic and initial issue seed definitions.
- `implementation_steps.json`: ordered execution tracker tied back to the roadmap documents.

## Applying the Tracker on GitHub

Use the `Sync Project Planning` workflow to create or update labels, milestones, and seed
issues from these files. The board configuration is recorded here for manual application or
for future PAT-backed automation because GitHub Project creation is not available from this
local workspace.
