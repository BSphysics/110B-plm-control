

import numpy as np

def super_pixel_set_init(phase_main , sPix_modes ):

    phase_main_expanded = np.expand_dims(phase_main*0.08, axis=-1)
    phase_zeros = np.zeros((phase_main.shape[0] ,phase_main.shape[1], 1))
    
    alternating_slices = np.concatenate(
    [phase_main_expanded if i % 2 == 0 else phase_zeros for i in range(24)], 
    axis=2)
    
    sPix_modes_init = np.concatenate((alternating_slices , sPix_modes) , axis = 2)
    
    

    sPix_modes_init = sPix_modes_init.astype('float32')

    return sPix_modes_init