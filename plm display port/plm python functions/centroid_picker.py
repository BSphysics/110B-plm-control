# -*- coding: utf-8 -*-
"""
centroid_picker.py

Opens a summed image from a .npy stack and lets you click the centroids
of each spot. Saves the result as  <same_folder>/<basename>_centroids.npy

Usage:
    Run this script directly. A file dialog will open.
    Click each spot centroid in order (left to right, or whatever order
    your analysis script expects).
    Press  Enter / Return  to confirm and save.
    Press  Backspace       to undo the last click.
    Press  Escape          to cancel without saving.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import os

scriptDir = os.getcwd()

# ── how many spots to pick ──────────────────────────────────────────────────
N_SPOTS = 7

# ── colour scheme ───────────────────────────────────────────────────────────
COL_RETICLE  = "#00ffcc"
COL_LABEL    = "#ffdd00"
COL_STATUS_OK  = "#00cc66"
COL_STATUS_BAD = "#ff4444"
COL_STATUS_WAIT = "#aaaaaa"
BG_DARK      = "#1a1a2e"


def pick_centroids(file_path: str, n_spots: int = N_SPOTS) -> np.ndarray | None:
    """
    Interactive centroid picker.

    Parameters
    ----------
    file_path : str   Path to a .npy image stack  (N, H, W)
    n_spots   : int   Number of spots to pick

    Returns
    -------
    centroids : np.ndarray, shape (n_spots, 2), columns = [x, y]  or None
    """

    imgs = np.load(file_path)
    img_sum = np.sum(imgs, axis=0).astype(np.float32)

    # ── state ──────────────────────────────────────────────────────────────
    clicks   = []          # list of (x, y)
    artists  = []          # matplotlib artists to remove on undo
    done     = [False]
    cancelled = [False]

    # ── figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    fig.subplots_adjust(bottom=0.14)

    ax.imshow(img_sum, cmap="inferno", interpolation="nearest")
    ax.set_title(
        f"Click {n_spots} spot centroids  |  Backspace = undo  |  Enter = confirm  |  Esc = cancel",
        color="white", fontsize=9, pad=8
    )
    ax.axis("off")

    # ── status text ────────────────────────────────────────────────────────
    status_txt = fig.text(
        0.5, 0.04,
        f"Clicks: 0 / {n_spots}",
        ha="center", va="center",
        fontsize=11, color=COL_STATUS_WAIT,
        fontweight="bold"
    )

    def refresh_status():
        n = len(clicks)
        if n < n_spots:
            status_txt.set_text(f"Clicks: {n} / {n_spots}  — keep clicking")
            status_txt.set_color(COL_STATUS_WAIT)
        else:
            status_txt.set_text(f"All {n_spots} spots picked — press Enter to save or Backspace to undo")
            status_txt.set_color(COL_STATUS_OK)
        fig.canvas.draw_idle()

    # ── buttons ────────────────────────────────────────────────────────────
    ax_confirm = fig.add_axes([0.60, 0.01, 0.17, 0.05])
    ax_undo    = fig.add_axes([0.40, 0.01, 0.17, 0.05])
    ax_cancel  = fig.add_axes([0.20, 0.01, 0.17, 0.05])

    btn_confirm = Button(ax_confirm, "Confirm (Enter)",  color="#224422", hovercolor="#336633")
    btn_undo    = Button(ax_undo,    "Undo (Backspace)", color="#443322", hovercolor="#665533")
    btn_cancel  = Button(ax_cancel,  "Cancel (Esc)",     color="#442222", hovercolor="#663333")

    for btn in (btn_confirm, btn_undo, btn_cancel):
        btn.label.set_color("white")
        btn.label.set_fontsize(8)

    # ── click handler ──────────────────────────────────────────────────────
    def on_click(event):
        if event.inaxes is not ax:
            return
        if event.button != 1:
            return
        if len(clicks) >= n_spots:
            return

        x, y = event.xdata, event.ydata
        clicks.append((x, y))
        idx = len(clicks)

        # reticle: cross + circle
        r = 7
        cross_h, = ax.plot([x - r, x + r], [y,     y    ], color=COL_RETICLE, lw=1.2)
        cross_v, = ax.plot([x,     x    ], [y - r, y + r], color=COL_RETICLE, lw=1.2)
        circ = patches.Circle((x, y), radius=r, fill=False,
                               edgecolor=COL_RETICLE, lw=1.5, linestyle="--")
        ax.add_patch(circ)
        lbl = ax.text(x + r + 2, y - r - 2, str(idx),
                      color=COL_LABEL, fontsize=9, fontweight="bold")
        artists.append((cross_h, cross_v, circ, lbl))

        refresh_status()

    # ── confirm ────────────────────────────────────────────────────────────
    def confirm(_=None):
        if len(clicks) != n_spots:
            status_txt.set_text(f"Need exactly {n_spots} clicks — you have {len(clicks)}")
            status_txt.set_color(COL_STATUS_BAD)
            fig.canvas.draw_idle()
            return
        done[0] = True
        fig.canvas.draw_idle()
        plt.close(fig)

    # ── undo ───────────────────────────────────────────────────────────────
    def undo(_=None):
        if not clicks:
            return
        clicks.pop()
        grp = artists.pop()
        for a in grp:
            a.remove()
        refresh_status()

    # ── cancel ─────────────────────────────────────────────────────────────
    def cancel(_=None):
        cancelled[0] = True
        plt.close(fig)

    # ── keyboard: re-focus the axes on every key event so Enter always works ──
    def on_key(event):
        # Print key name to console so you can diagnose any remaining issues
        print(f"[key pressed: {repr(event.key)}]")
        k = (event.key or "").lower().strip()
        if k in ("enter", "return", "\n", "\r", "ctrl+m"):
            confirm()
        elif k == "backspace":
            undo()
        elif k in ("escape", "q"):
            cancel()

    # ── click on the axes re-grabs focus so keyboard works after button clicks ──
    def on_axes_click(event):
        if event.inaxes is ax:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_press_event", on_axes_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_confirm.on_clicked(confirm)
    btn_undo.on_clicked(undo)
    btn_cancel.on_clicked(cancel)

    refresh_status()

    # Use a blocking show loop so we can check state after the window closes
    plt.show(block=True)

    if cancelled[0] or not done[0]:
        print("Centroid picking cancelled — no file saved.")
        return None

    return np.array(clicks, dtype=np.float32)   # shape (N, 2): columns = x, y


def main():
    # ── file dialog ────────────────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select .npy image stack",
        filetypes=[("NumPy files", "*.npy")]
    )
    root.destroy()

    if not file_path:
        print("No file selected.")
        return

    print(f"\nLoading: {file_path}")

    centroids = pick_centroids(file_path, n_spots=N_SPOTS)

    if centroids is None:
        return

    # ── save alongside the data file ───────────────────────────────────────
    # base = os.path.splitext(file_path)[0]
    save_path = os.path.join(scriptDir, "plough_centroids.npy")
    # save_path = base + "_centroids.npy"
    np.save(save_path, centroids)

    print(f"\nSaved {N_SPOTS} centroids to:\n  {save_path}")
    print("\nCentroids (x, y):")
    for i, (x, y) in enumerate(centroids):
        print(f"  Spot {i+1}: x={x:.1f}, y={y:.1f}")


if __name__ == "__main__":
    main()