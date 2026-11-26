"""
elliptec.py - one driver for all the Thorlabs Elliptec devices on the PLM rig:
    ELL14 rotation mounts (linear polariser, QWP) and the ELL slider (ND filter).

    from elliptec import ELL14, ELLSlider, close_all
    pol    = ELL14(serial_no="11400476")       # polariser  (currently COM5, address 1)
    qwp    = ELL14(serial_no="11402101")       # QWP        (currently COM6, address 0)
    slider = ELLSlider(port="COM4", address="0")   # or serial_no=... once you know it
    pol.home(); pol.move_abs(50.3)
    slider.forward(); slider.backward()
    close_all()                                # e.g. in the GUI's closeEvent

Design points
- Devices can be found by their device serial number, so COM renumbering doesn't
  silently swap the polariser and QWP.
- Every motion command blocks until the device replies with its final position
  (or raises on an error status) instead of sleeping a fixed time.
- Ports are opened on demand and closed with release(), so legacy code that opens
  the same COM port itself (e.g. move_slider_to_not_attenuator) still works as long
  as you release() before calling it.  Two devices on one bus share one Serial object.
"""
import time
import serial
from serial.tools import list_ports

_PORTS = {}

MODELS = {0x06: "ELL6 2-position slider", 0x09: "ELL9 4-position slider",
          0x0E: "ELL14 rotation mount", 0x12: "ELL18 rotation stage",
          0x14: "ELL20 linear stage", 0x11: "ELL17 linear stage"}

STATUS = {"00": "OK", "01": "communication timeout", "02": "mechanical timeout",
          "03": "command not supported", "04": "value out of range",
          "05": "module isolated", "06": "module out of isolation",
          "07": "initialising error", "08": "thermal error", "09": "busy",
          "0A": "sensor error", "0B": "motor error", "0C": "out of range",
          "0D": "over current"}


def _open(port):
    s = _PORTS.get(port)
    if s is None or not s.is_open:
        s = serial.Serial(port, 9600, bytesize=8, parity="N", stopbits=1, timeout=0.1)
        time.sleep(0.05)
        s.reset_input_buffer()
        _PORTS[port] = s
    return s


def release(port):
    s = _PORTS.pop(port, None)
    if s is not None and s.is_open:
        s.close()


def close_all():
    for p in list(_PORTS):
        release(p)


def parse_info(reply):
    """Decode an 'IN' reply, e.g. '1IN0E1140047620221601016800023000'."""
    r = reply.strip()
    model = int(r[3:5], 16)
    return dict(address=r[0], model=model, model_name=MODELS.get(model, f"type 0x{model:02X}"),
                serial=r[5:13], year=int(r[13:17]), firmware=r[17:19], hardware=r[19:21],
                travel=int(r[21:25], 16), pulses_per_unit=int(r[25:33], 16))


def _int32(h):
    v = int(h, 16)
    return v - (1 << 32) if v >= (1 << 31) else v


def _query(ser, cmd, timeout=1.0):
    ser.reset_input_buffer()
    ser.write(cmd.encode("ascii"))
    t0, buf = time.time(), b""
    while time.time() - t0 < timeout:
        buf += ser.read(64)
        if buf.endswith(b"\r\n"):
            break
    return buf.decode("ascii", "replace").strip() or None


def scan(ports=None, addresses="0123456789ABCDEF"):
    """List every Elliptec device found: [(port, address, info), ...]."""
    found = []
    for port in ports or [p.device for p in list_ports.comports()]:
        try:
            ser = _open(port)
        except (serial.SerialException, OSError):
            continue
        for a in addresses:
            rep = _query(ser, f"{a}in", timeout=0.3)
            if rep and rep[1:3] == "IN" and len(rep) >= 33:
                found.append((port, a, parse_info(rep)))
        release(port)
    return found


class ELLDevice:
    def __init__(self, serial_no=None, port=None, address=None):
        if serial_no is not None:
            hits = [h for h in scan() if h[2]["serial"] == serial_no]
            if not hits:
                raise RuntimeError(f"Elliptec device with serial {serial_no} not found")
            port, address, info = hits[0]
        else:
            if port is None or address is None:
                raise ValueError("give serial_no, or port and address")
            rep = _query(_open(port), f"{address}in")
            if not rep or rep[1:3] != "IN":
                raise RuntimeError(f"no Elliptec device at {port} address {address} (got {rep!r})")
            info = parse_info(rep)
        self.port, self.addr, self.info = port, str(address), info

    def __repr__(self):
        return (f"{type(self).__name__}({self.info['model_name']}, serial={self.info['serial']}, "
                f"{self.port}, addr {self.addr})")

    @property
    def ser(self):
        return _open(self.port)

    def release(self):
        """Close this device's COM port (other code may then open it)."""
        release(self.port)

    def _wait(self, timeout):
        t0, buf = time.time(), b""
        ser = self.ser
        while time.time() - t0 < timeout:
            buf += ser.read(64)
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                s = line.decode("ascii", "replace")
                if not s.startswith(self.addr):
                    continue
                if s[1:3] == "PO":
                    return _int32(s[3:11])
                if s[1:3] == "GS" and s[3:5] not in ("00", "09"):
                    raise RuntimeError(f"{self}: {STATUS.get(s[3:5], s)}")
        raise TimeoutError(f"{self}: no position reply within {timeout} s")

    def _move(self, cmd, timeout=10.0):
        ser = self.ser
        ser.reset_input_buffer()
        ser.write(f"{self.addr}{cmd}".encode("ascii"))
        return self._wait(timeout)

    def home(self, ccw=False):
        return self._move(f"ho{1 if ccw else 0}", timeout=15.0)

    def position_raw(self):
        return self._move("gp", timeout=1.0)

    def forward(self):
        return self._move("fw")

    def backward(self):
        return self._move("bw")

    def change_address(self, new):
        """Re-address the device (e.g. QWP 0 -> 2). Check it persists after a power cycle."""
        rep = _query(self.ser, f"{self.addr}ca{new}")
        self.addr = str(new)
        return rep


class ELL14(ELLDevice):
    """Rotation mount; angles in degrees."""

    @property
    def ppr(self):
        return self.info["pulses_per_unit"]          # 143360 for the ELL14

    def _hex(self, deg):
        return format(int(round(deg / 360 * self.ppr)) & 0xFFFFFFFF, "08X")

    def home(self, ccw=False):
        return super().home(ccw) / self.ppr * 360

    def move_abs(self, deg):
        return self._move("ma" + self._hex(deg % 360)) / self.ppr * 360

    def move_rel(self, deg):
        return self._move("mr" + self._hex(deg)) / self.ppr * 360

    def position(self):
        return self.position_raw() / self.ppr * 360


class ELLSlider(ELLDevice):
    """Multi-position slider. Positions are reported in raw pulses; which one is
    'Attenuator' is learnt from the position the slider reports after a move."""
    pass
