# %%
########start

# %%
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

# %%
import os
import mne

directory = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg/"
filename = "sub-01_task-RDR_run-37_meg.fif"

# Define the file path
file_path = os.path.join(directory, filename)

# Load raw data
raw = mne.io.read_raw_fif(file_path)

# Perform additional preprocessing as needed: artifact removal, downsampling, etc.


# %%
channels_to_remove = ['IASX-', 'IAS_DY', 'STI005', 'STI008', 'IASY-', 'IAS_DX', 'STI007', 'SYS201', 'IAS_X', 'BIO001', 'IASY+', 'IAS_Y', 'STI002', 'STI003', 'IASX+', 'IASZ+', 'STI006', 'STI101', 'STI004', 'IAS_Z', 'STI001', 'IASZ-']

raw = raw.pick_channels(ch_names=[ch for ch in raw.ch_names if ch not in channels_to_remove])


# %%


# %%
# Specify the available channels in the picks parameter
#picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True)  # Adjust the channel types as per your data

stim_channel = raw.info["ch_names"][picks[0]]  # Extract the stim channel name

events = mne.find_events(raw, stim_channel=stim_channel)
if len(events) > 0:
    event_id = dict(aud_l=1)  # event trigger and conditions
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)
    raw.info["bads"] = ["MEG 2443", "EEG 053"]
    baseline = (None, 0)  # means from the first instant to t = 0
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=("meg", "eog"),
        baseline=baseline,
        reject=reject,
    )
else:
    print("No events found. Cannot create epochs.")



# %%


# %%
# Compute covariance matrix for resting-state analysis
noise_cov = mne.compute_raw_covariance(raw)

# %%
# Plot covariance matrix
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)


# %%
import mne

# Set the directory and filename
#directory = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg/"
#filename = "sub-01_task-RDR_run-37_meg.fif"

# Load the raw data
#raw = mne.io.read_raw_fif(directory + filename, preload=True)

# Create an info object with the necessary information
info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=raw.info['sfreq'], ch_types='grad')

# Convert the raw data to an Evoked object
evoked = mne.EvokedArray(data=raw.get_data(), info=info)

# Plot the evoked response
evoked.plot(time_unit="s")


# %%
start_time = 0.5  # Start time in seconds
end_time = 1.5  # End time in seconds
evoked_crop = evoked.crop(tmin=start_time, tmax=end_time)
evoked_crop.plot()


# %%
evoked.info

# %%


# %%
common_channels = list(set(evoked.ch_names) & set(noise_cov.ch_names))
noise_cov = noise_cov.pick_channels(common_channels)


# %%
evoked.plot_white(noise_cov, time_unit="s")
#del epochs, raw  # to save memory

# %%
import os

directory = "/data/users2/mjafarlou1/results/forwardsolution/sub-01"
filename = "sub-01_task-RDR_run-37_meg_fwd.fif"

filepath = os.path.join(directory, filename)
fwd = mne.read_forward_solution(filepath)


# %%
fwd.values()

# %%
raw_channels = raw.ch_names
forward_channels = fwd['info']['ch_names']


# %%
fwd.keys()

# %%
plt.figure(figsize=(15,20))
plt.imshow(fwd['sol']['data'],aspect=30, interpolation=None)

# %%
plt.plot(fwd['sol']['data'].sum(axis=1))

# %%
fwd['nsource']*3

# %%
# Assuming you have the necessary imports and dependencies for `make_inverse_operator`
inverse_operator = mne.minimum_norm.make_inverse_operator( # insted of make_inverse_operator
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
)
del fwd

# %%
inverse_operator['eigen_fields']['data']

# %%
plt.imshow(inverse_operator['eigen_fields']['data'], aspect=1, interpolation=None, cmap=plt.cm.RdBu_r)
plt.colorbar()

# %%
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2

stc, residual = mne.minimum_norm.apply_inverse(
    evoked,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# %%


# %%



