import re
import os

docs_path = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/docs/FLUXFORGE_CONSOLIDATED_MASTER.md"
with open(docs_path, "r") as f:
    content = f.read()

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"
os.makedirs(issues_dir, exist_ok=True)

planned_lines = [line for line in content.split("\n") if "| Planned |" in line]

count = 0
for line in planned_lines:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) >= 3:
        issue_id = parts[0]
        title = parts[1]

        filename = f"{issue_id}_{re.sub(r'[^A-Za-z0-9]', '_', title)}.md"
        filepath = os.path.join(issues_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"# Issue: {issue_id} - {title}\n\n")
            f.write(f"**Status:** Planned\n")
            for j, part in enumerate(parts[2:]):
                if part != "Planned":
                    f.write(f"**Context:** {part}\n")
            f.write(f"\n## Description\n")
            f.write(
                f"This issue is generated from the master plan and needs implementation.\n"
            )
        count += 1

print(f"Created {count} planned issues.")
