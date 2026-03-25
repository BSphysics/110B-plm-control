# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:15:07 2025

@author: bs426
"""

import numpy as np
from numba import njit

@njit
def super_pixel_set_numba(sPix_W, sPix_H, phase_main, n_steps):
    H, W = phase_main.shape

    nx_sPix = (W + sPix_W - 1) // sPix_W
    ny_sPix = (H + sPix_H - 1) // sPix_H
    n_sPix = nx_sPix * ny_sPix

    phase_step = 1.0 / n_steps
    n_modes = n_sPix * (n_steps + 1)

    i_ref = ny_sPix // 2
    j_ref = nx_sPix // 2
    row_idx_ref = i_ref * sPix_H
    col_idx_ref = j_ref * sPix_W

    ref_sPix_phase = phase_main[row_idx_ref:row_idx_ref + sPix_H,
                                col_idx_ref:col_idx_ref + sPix_W]

    sPix_modes = np.zeros((H, W, n_modes), dtype=np.float32)
    page_counter = 0

    for i in range(ny_sPix):
        for j in range(nx_sPix):
            row_idx = i * sPix_H
            col_idx = j * sPix_W
            page_idx = page_counter * (n_steps + 1)

            test_sPix_phase = phase_main[row_idx:row_idx + sPix_H,
                                         col_idx:col_idx + sPix_W]

            sPix_modes[row_idx:row_idx + sPix_H,
                       col_idx:col_idx + sPix_W,
                       page_idx] = test_sPix_phase

            for k in range(n_steps):
                stepped_phase = (test_sPix_phase + k * phase_step) % 1.0

                sPix_modes[row_idx:row_idx + sPix_H,
                           col_idx:col_idx + sPix_W,
                           page_idx + k + 1] = stepped_phase

                sPix_modes[row_idx_ref:row_idx_ref + sPix_H,
                           col_idx_ref:col_idx_ref + sPix_W,
                           page_idx + k + 1] = ref_sPix_phase

            page_counter += 1


    return sPix_modes, nx_sPix, ny_sPix