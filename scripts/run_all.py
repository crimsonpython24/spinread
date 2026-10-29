import os
import subprocess
import sys

STEPS = ["prepare_data", "build_cells", "run_models", "error_analysis"]

for name in STEPS:
    print(f"\n===== {name} =====", flush=True)
    subprocess.run(
        [sys.executable, os.path.join("scripts", f"{name}.py")],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        check=True,
    )
