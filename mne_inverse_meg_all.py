##coregistration

from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.coreg import Coregistration
from mne.io import read_info

sub01_fid = np.array([[-72.90, -2.57, -63.73],[-0.49, 87.38, -53.45],[75.54, -3.75, -63.74]]) #LPA, Nasion, RPA

# Set the paths
source_dir = '/data/users2/nshor/Multiband_with_MEG/'
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects/'
coreg_dir = '/data/users2/mjafarlou1/freesurfer/subjects/sub-01/coreg/'
raw_directory = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg/"
fwd_directory = "/data/users2/mjafarlou1/results/forwardsolution/sub-01"
stc_directory = "/data/users2/mjafarlou1/results/stc/sub-01"

if not os.path.exists(fwd_directory):
    os.makedirs(fwd_directory)

if not os.path.exists(stc_directory):
    os.makedirs(stc_directory)

if not os.path.exists(coreg_dir):
    os.makedirs(coreg_dir)

# set the SUBJECTS_DIR environment variable
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects'
os.environ['SUBJECTS_DIR'] = subjects_dir

if not os.path.exists(coreg_dir):
    os.makedirs(coreg_dir)

subject = 'sub-01'

for sess in range(60):
    sess = str(sess + 1)
    print(sess)
    fname_raw = os.path.join(source_dir, subject, 'meg', f'{subject}_task-RDR_run-{sess}_meg.fif')
    print(fname_raw)
    info = read_info(fname_raw)
    plot_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                       surfaces="head-dense", dig=True, eeg=[],
                       meg='sensors', show_axes=True,
                       coord_frame='meg')
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                       focalpoint=(0., 0., 0.))

    fiducials = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
    sub_fid = 'sub' + subject[3:] + '_fid'
    new_id = sub_fid.replace('-', '')
    for i in range(3):
        fiducials[i]['r'] = locals()[new_id][i, :] * 0.001

    coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    coreg.fit_fiducials(verbose=True)

    coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True)
    coreg.omit_head_shape_points(distance=5. / 1000)

    coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3
    print(f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
          f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm")

    out_path = os.path.join(coreg_dir, '')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    mne.write_trans(out_path + f'/{subject}_task-RDR_run-{sess}_trans.fif', coreg.trans, overwrite=True)
    coreg.trans

coreg=None
    
#Compute source spaces and Compute forward solution

# Load source space
src = mne.setup_source_space(subject, spacing='all', n_jobs=-1, add_dist='patch', subjects_dir=subjects_dir)

plot_bem_kwargs = dict(
    brain_surfaces='white',  # Specify the color of the brain surfaces
    src=src,  # Provide the source space
    show=True  # Display the plot
)

mne.viz.plot_bem(subject=subject, **plot_bem_kwargs)

# Compute BEM model and solution
conductivity = (0.3,)  # for single layer
model = mne.make_bem_model(subject="sub-01", ico=None, conductivity=conductivity, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Loop through raw files in the directory
for filename in os.listdir(raw_directory):
    if filename.endswith(".fif"):
        raw_fname = os.path.join(raw_directory, filename)
        
        # Find corresponding transformation file
        trans_filename = f"{filename[:-8]}_trans.fif"
        trans_fname = os.path.join(coreg_dir, trans_filename)
        
        # Read transformation file
        trans = mne.read_trans(trans_fname)
        
        # Compute forward solution
        fwd = mne.make_forward_solution(
            raw_fname,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=-1,
            verbose=True,
        )
        
        print(fwd)
        
        # convert the forward model with free source orientation to a model 
        # with orientation fixed to the direction orthogonal to each vertex 
        # on the surface
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, 
                                           force_fixed=True, copy=False, 
                                           use_cps=True,
                                           verbose=None)
        
        raw = mne.io.read_raw_fif(raw_fname, preload=False)
        num_raw_channels = len(raw.info['ch_names'])
        print("Number of channels in raw data:", num_raw_channels)
        
        num_forward_channels = len(fwd['info']['ch_names'])
        print("Number of channels in the forward solution:", num_forward_channels)
        
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        forward_channels = fwd['info']['ch_names']

        raw_channels = raw.info['ch_names']
        missing_channels = set(raw_channels) - set(forward_channels)
        
        print("Missing channels in the forward solution:", missing_channels)
        
        # Save the forward solution
        output_file_path = os.path.join(fwd_directory, f"{filename[:-8]}_fwd.fif")
        mne.write_forward_solution(output_file_path, fwd, overwrite=True)
        
        print(f"Forward solution saved for {filename}")

data_files = [file for file in os.listdir(raw_directory) if file.endswith('.fif')]

# Iterate over each data file
for data_file in data_files:
    filename = os.path.join(raw_directory, data_file)
    raw = mne.io.read_raw_fif(filename)  # already has an average reference

    # Specify the available channels in the picks parameter
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
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

    channels_to_remove = ['IASX-', 'IAS_DY', 'STI005', 'STI008', 'IASY-', 'IAS_DX', 'STI007', 'SYS201', 'IAS_X', 'BIO001', 'IASY+', 'IAS_Y', 'STI002', 'STI003', 'IASX+', 'IASZ+', 'STI006', 'STI101', 'STI004', 'IAS_Z', 'STI001', 'IASZ-']

    raw = raw.pick_channels(ch_names=[ch for ch in raw.ch_names if ch not in channels_to_remove])

    # compute covariance matrix for resting-state analysis
    noise_cov = mne.compute_raw_covariance(raw)

    # plot covariance matrix
    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

    # Create an info object with the necessary information
    info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=raw.info['sfreq'], ch_types='grad')

    # Convert the raw data to an Evoked object
    evoked = mne.EvokedArray(data=raw.get_data(), info=info)

    # Plot the evoked response
    evoked.plot(time_unit="s")

    common_channels = list(set(evoked.ch_names) & set(noise_cov.ch_names))
    noise_cov = noise_cov.pick_channels(common_channels)

    evoked.plot_white(noise_cov, time_unit="s")

    start_time = 0  # Start time in seconds
    end_time = 5  # End time in seconds
    evoked_crop = evoked.crop(tmin=start_time, tmax=end_time)
    evoked_crop.plot()

    # Load the forward solution for the corresponding data file
    fwd_filename = os.path.join(fwd_directory, data_file.replace('.fif', '_fwd.fif'))
    fwd = mne.read_forward_solution(fwd_filename)

    raw_channels = raw.ch_names
    forward_channels = fwd['info']['ch_names']

    fwd['nsource'] * 3

    inverse_operator = mne.minimum_norm.make_inverse_operator(  # instead of make_inverse_operator
        evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
    )
    del fwd

    inverse_operator['eigen_fields']['data']

    method = "dSPM"
    snr = 2.0
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

    # Save the results

    stc_filename = os.path.join(stc_directory , data_file.replace('.fif', '.stc'))
    residual_filename = os.path.join(stc_directory , data_file.replace('.fif', '_residual.stc'))
    stc.save(stc_filename, overwrite=True)
    residual.save(residual_filename, overwrite=True)
