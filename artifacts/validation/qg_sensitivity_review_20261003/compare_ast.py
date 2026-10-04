import ast
import subprocess
old='7c5cc8b'; new='be4e7f5'
files={'src/fluxforge/examples/rafm_workflow.py':['default_paths','load_rafm_example_metadata','workflow_profile_energy_calibration','build_generic_gamma_library','discover_input_files','pair_input_files','select_generic_targeted_lines','merge_detected_and_targeted_peaks','qg_reference_peaks','match_peak_set','resolve_measurement_timing','normalize_pairing_key'], 'src/fluxforge/analysis/flux_wire_analysis.py':['analyze_raw_spectrum','analyze_raw_spectrum_targeted','build_gamma_library','get_expected_isotopes'], 'src/fluxforge/io/flux_wire.py':['read_raw_asc','read_processed_txt'], 'src/fluxforge/analysis/spectrum_math.py':['subtract_measured_background']}
def get(rev,path):
    try: return subprocess.check_output(['git','show',f'{rev}:{path}']).decode('utf-8-sig')
    except subprocess.CalledProcessError: return None
def functions(src):
    if src is None:return {}
    tree=ast.parse(src)
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for path,names in files.items():
    a,b=functions(get(old,path)),functions(get(new,path))
    print(f'FILE {path} old_present={bool(a)} new_present={bool(b)}')
    for name in names:
        print(f'  {name}: {"IDENTICAL" if name in a and name in b and a[name]==b[name] else "CHANGED/MISSING"} old={name in a} new={name in b}')
