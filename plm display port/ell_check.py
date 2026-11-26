"""
ell_check.py - does reconnecting from a NEW Python process change the ELL14s' reported position?

Run from the GUI's folder (D:\\PLM\\plm display port), with the GUI closed:
    python ell_check.py move      # moves both stages to 100 deg and reads them back
    python ell_check.py read      # (second, separate process) reads them WITHOUT moving
Both lines should print 100.00 for both stages.
"""
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "plm python functions"))
from elliptec import ELL14, close_all

mode = sys.argv[1] if len(sys.argv) > 1 else "read"
pol = ELL14(port="COM5", address="1")
qwp = ELL14(port="COM6", address="0")
if mode == "move":
    pol.move_abs(100)
    qwp.move_abs(100)
print(f"[{mode}] polariser {pol.position():8.3f} deg   QWP {qwp.position():8.3f} deg")
close_all()
