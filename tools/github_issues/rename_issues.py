import os
import re

issues_dir = "/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples/RAFM_irradiation/issues"
files = sorted(os.listdir(issues_dir))

# Find the highest numbered prefix
max_num = 0
for f in files:
    match = re.match(r"^(\d+)_", f)
    if match:
        max_num = max(max_num, int(match.group(1)))

next_num = max_num + 1

for f in files:
    # Skip already numbered files
    if re.match(r"^\d+_", f):
        continue
    
    if not f.endswith('.md'):
        continue

    # Rename file
    old_path = os.path.join(issues_dir, f)
    new_name = f"{next_num:02d}_{f}"
    new_path = os.path.join(issues_dir, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed {f} -> {new_name}")
    next_num += 1

print(f"Renamed up to {next_num - 1}")
