import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


roi_sums = np.load('roi_sums.npy')

#print(roi_sums)
x_data  = np.linspace(0, 2*np.pi , 10)

def cos_squared_model(theta, Ic, Is, theta_0 ):
    return Ic * (np.cos(0.5*theta + theta_0))**2 + Is * np.sin(0.5*theta + theta_0)**2

# Initial guesses: 
guess = [np.ptp(roi_sums)/2, np.ptp(roi_sums)/2 , np.pi]
params, _ = curve_fit(cos_squared_model, x_data, roi_sums, p0=guess)
Ic_fit, Is_fit , theta0_fit  = params

#print('Zero Phase (%) = ' + str(np.round(theta0_fit / (2*np.pi) *100)))

# Generate smooth fit curve
theta_fine = np.linspace(0, 2 * np.pi, 500)
fit_curve = cos_squared_model(theta_fine, Ic_fit, Is_fit , theta0_fit)



# Plot
plt.close('all')
plt.figure(figsize=(8, 4))
plt.plot(x_data, roi_sums, 'o', label='Data points')  # points only
plt.plot(theta_fine, fit_curve, '-', label=r'Fit: $I_0cos^2(\theta + \theta_0)$')
plt.xlabel('θ (radians)')
plt.ylabel('Intensity')
plt.title('Cos² Fit to ROI Sums')
plt.show()

idx_min = np.argmin(fit_curve)

print('Zero Phase (%) = ' + str(np.round(theta_fine[idx_min] / (2*np.pi) *100 , 1)))
