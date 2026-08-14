import mne
import numpy as np
import matplotlib.pyplot as plt

valid_sensors = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8']
valid_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5])

info = mne.create_info(ch_names=valid_sensors, sfreq=100, ch_types="eeg")
info.set_montage("standard_1020", match_case=False)

fig, ax = plt.subplots()
im, _ = mne.viz.plot_topomap(valid_data, info, axes=ax, show=False, vlim=(0.0, 1.0))
cbar = plt.colorbar(im, ax=ax)
fig.savefig("test_vlim.png")
print(f"clim is {im.get_clim()}")
