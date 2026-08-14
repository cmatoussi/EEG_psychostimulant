import numpy as np
import mne

info = mne.create_info(ch_names=['Cz', 'Pz', 'Fz', 'C3', 'C4'], sfreq=100, ch_types='eeg')
info.set_montage('standard_1020')
try:
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks="all")
except ImportError:
    pos = mne.viz.topomap._find_topomap_coords(info, picks="all")
    
print("Coords min max:", pos.min(), pos.max())
