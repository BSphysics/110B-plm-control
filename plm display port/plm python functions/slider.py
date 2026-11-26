"""
slider.py - ND-filter slider toggle for the PLM GUI, using elliptec.py.

Same interface as before:
    slider(self)                        toggle (first press initialises to 'Attenuator')
    move_slider_to_not_attenuator(self) ensure the ND filter is out of the beam
GUI attributes used: self.serial_port, self.slider_position, self.slider_button,
self.update_slider_button_style().  Optional: self.slider_serial_no (device serial;
if set, the slider is found by serial instead of COM port) and self.slider_address
(default '0').

Slider on this rig: ELL6 2-position, serial 10600312, COM4, address 0.

Differences from the old version
- The state is set from the OUTCOME of each move (completed, or refused because the
  slider is already at that end), not assumed.  If a move fails, the state becomes unknown (None) and the next press
  re-initialises, so the button colour can't drift out of step with the hardware.
- Moves wait for the slider's reply instead of sleep(1.5).
- The COM port is released after every operation, as before, so other code can open it.
"""
from elliptec import ELLSlider

ATTENUATOR_CMD = "forward"        # old code: '0fw' = Attenuator, '0bw' = No attenuator


def _device(self):
    dev = getattr(self, "_slider_dev", None)
    if dev is None:
        sn = getattr(self, "slider_serial_no", None)
        if sn:
            dev = ELLSlider(serial_no=sn)
        else:
            dev = ELLSlider(port=self.serial_port, address=getattr(self, "slider_address", "0"))
        print(f"Slider: {dev}")
        self._slider_dev = dev
    return dev


# Status codes meaning "can't go further that way": the slider is already at that end.
_AT_END = ("out of range", "value out of range")


def _go(self, dev, attenuator):
    """Move to the attenuator end (forward) or the clear end (backward) and return the
    resulting state.  The ELL6 reports 0 pulses/unit, so the state is taken from the
    OUTCOME of the move, not from position arithmetic:
      - move completes (PO reply)                -> at the requested end
      - refused as out of range                  -> already at the requested end
      - any other error (e.g. mechanical timeout)-> read the position and compare it
        with the positions previously seen at each end; unknown (None) if we can't tell
    """
    target = "Attenuator" if attenuator else "No attenuator"
    fwd = attenuator == (ATTENUATOR_CMD == "forward")
    seen = self.__dict__.setdefault("_slider_seen_pos", {})
    try:
        pos = dev.forward() if fwd else dev.backward()
        seen[target] = pos
        return target
    except RuntimeError as e:
        if any(k in str(e) for k in _AT_END):
            return target
        print(f"  slider: {e}")
        try:
            pos = dev.position_raw()
        except Exception:                                       # noqa: BLE001
            return None
        for state, p in seen.items():
            if p == pos:
                return state
        return None


def _set_position(self, attenuator):
    dev = None
    try:
        dev = _device(self)
        self.slider_position = _go(self, dev, attenuator)
        if self.slider_position is None:
            print("Slider state unknown - next press will re-initialise")
    except Exception as e:                                      # noqa: BLE001
        print(f"Slider error: {e} - state unknown, next press will re-initialise")
        self.slider_position = None
        self._slider_dev = None
    finally:
        if dev is not None:
            dev.release()


def slider(self):
    """Toggle the ND filter in/out of the beam."""
    self.slider_button.setEnabled(False)
    try:
        if self.slider_position is None:
            print("Initialising slider: forcing to attenuator position..")
            _set_position(self, attenuator=True)
        else:
            want_att = self.slider_position != "Attenuator"
            print(f"Moving slider: {self.slider_position} -> "
                  f"{'Attenuator' if want_att else 'No attenuator'}")
            _set_position(self, attenuator=want_att)
        print(f"Slider is now at position {self.slider_position}")
    finally:
        self.update_slider_button_style()
        self.slider_button.setEnabled(True)


def move_slider_to_not_attenuator(self):
    """Take the ND filter out of the beam (no-op if already out)."""
    if self.slider_position != "No attenuator":
        _set_position(self, attenuator=False)
        print(f"Slider is now at position {self.slider_position}")
        self.update_slider_button_style()


def move_slider_to_attenuator(self):
    if self.slider_position != "Attenuator":
        _set_position(self, attenuator=True)
        print(f"Slider is now at position {self.slider_position}")
        self.update_slider_button_style()
