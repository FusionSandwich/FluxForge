"""Trace the bounded production-copy check without changing GUI or test code."""
import faulthandler
import importlib.util
import json
from pathlib import Path
import sys

out=Path(__file__).resolve().parent
repo=out.parents[2]
path=repo/"tests/test_production_gui_mode.py"
spec=importlib.util.spec_from_file_location("production_copy_probe",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stream=(out/"qt_copy_trace.jsonl").open("w",encoding="utf-8")
def trace(frame,event,arg):
    if frame.f_code.co_name == "test_production_visible_copy_contains_no_implementation_narration" and event=="line":
        stream.write(json.dumps({"line":frame.f_lineno})+"\n")
        stream.flush()
    return trace

with (out/"qt_copy_stack.txt").open("w") as stack:
    faulthandler.dump_traceback_later(25,file=stack)
    sys.settrace(trace)
    try:
        module.test_production_visible_copy_contains_no_implementation_narration()
        print("production copy check passed")
    finally:
        sys.settrace(None)
        faulthandler.cancel_dump_traceback_later()
        stream.close()
