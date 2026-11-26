"""
ell_set.py - set the analyser without homing (GUI can stay open; don't run a GUI
measurement at the same moment).

    python ell_set.py 173.7          polariser to 173.7 deg, QWP parallel (P + theta0 from analyser_cal.json)
    python ell_set.py 173.7 222.75   polariser and QWP to explicit stage angles
"""
import os
import sys
import json
sys.path.append(os.path.join(os.getcwd(), "plm python functions"))
from elliptec import ELL14, close_all

P = float(sys.argv[1])
if len(sys.argv) > 2:
    Q = float(sys.argv[2])
else:
    with open("analyser_cal.json") as f:
        Q = P + json.load(f)["theta0_deg"]
pol = ELL14(port="COM5", address="1")
qwp = ELL14(port="COM6", address="0")
print(f"polariser -> {pol.move_abs(P):.2f} deg, QWP -> {qwp.move_abs(Q):.2f} deg")
close_all()
