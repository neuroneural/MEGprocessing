import sys
import os
import pathlib
import os
import numpy as np
import matplotlib
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.coreg import Coregistration
from mne.io import read_info

def process_meg_data(source_dir, subjects_dir, stc_directory, subject):

    #coregistration
    fname_raw = source_dir
    info = read_info(fname_raw)

    fiducials = "estimated"  # get fiducials from fsaverage
    coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)

    coreg.fit_fiducials(verbose=True)

    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)

    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )
    
    raw = mne.io.read_raw_fif(source_dir, preload=True)

    info = raw.info
    trans = coreg.trans

    os.environ['SUBJECTS_DIR'] = subjects_dir

# Load source space
    spacing = 'ico5' # all-oct6
    src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir)

# Compute BEM model and solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    #  forward solution
    fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem, meg=True, eeg=False, ignore_ref=True,)

    fwd = mne.convert_forward_solution(fwd, surf_ori=True, 
                                           force_fixed=True, copy=False, 
                                           use_cps=True,
                                           verbose=None)

# covariance matrix
    cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method=['shrunk', 'diagonal_fixed', 'empirical'], rank='info')
    
# make inverse solution
    fixed = True  # set to False for free estimates
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, loose='auto', fixed=True)
    lambda2 = 1.0 / 2.0 ** 2 #SNR=2 for single trail
        
    stcs = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2, method='dSPM', use_cps=True)
    
    stcs
    raw_filename=source_dir

    stc_filename = os.path.join(stc_directory, os.path.basename(raw_filename).replace('.fif', '.stc'))
    stcs.save(stc_filename, overwrite=True)
    
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py /input_fif_file.fif/ /FreeSurfer_directory/ /output_directory/ /subject_ID/")
        sys.exit(1)

    subject_data = sys.argv[1]
    freesurfer_output_location = sys.argv[2]
    output_directory = sys.argv[3]
    subject = sys.argv[4]


    if not os.path.isfile(subject_data):
        print("Error: Input fif file not found.")
        sys.exit(1)

    if not os.path.isdir(freesurfer_output_location):
        print("Error: FreeSurfer directory not found.")
        sys.exit(1)

    if not os.path.isdir(output_directory):
        print("Error: Output directory not found.")
        sys.exit(1)

    process_meg_data(subject_data, freesurfer_output_location, output_directory, subject) 

