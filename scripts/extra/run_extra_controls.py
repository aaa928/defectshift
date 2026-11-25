"""Runner of the additional randomized lot-split controls: registers the additional control seeds
(defectshift.extra_controls.PLAN) in config.PLACEBO_B_TARGET and then runs the existing runner (04_tune_hyperparameters.py or
06_benchmark.py) unchanged. The shared configuration file is not changed.
Usage: python scripts/extra/run_extra_controls.py hpo <04_tune_hyperparameters.py arguments...>
       python scripts/extra/run_extra_controls.py bench <06_benchmark.py arguments...>
"""
import os
import runpy
import sys

from defectshift import extra_controls  # noqa: E402

TARGET = {"hpo": "04_tune_hyperparameters.py", "bench": "06_benchmark.py"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TARGET:
        raise SystemExit(f"usage: run_extra_controls.py {{{'|'.join(TARGET)}}} <runner arguments...>")
    which = sys.argv[1]
    extra_controls.register()
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), TARGET[which])
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
