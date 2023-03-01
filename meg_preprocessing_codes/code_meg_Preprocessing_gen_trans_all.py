import os.path as op
import numpy as np

import mne
from mne.coreg import Coregistration
from mne.io import read_info

# $ export SUBJECTS_DIR='/Volumes/Server/NEUROLING/PersonalFiles/Shaonan Wang/meg_pku/mri'
# $ mne coreg

ROOT = './'
subjects_dir = op.join(ROOT, 'meg_pku/mri')

sub01_fid = np.array([[-72.90, -2.57, -63.73],[-0.49, 87.38, -53.45],[75.54, -3.75, -63.74]]) #LPA, Nasion, RPA
sub02_fid = np.array([[-69.48, -2.53, -64.51],[0.49, 85.86, -51.75],[74.83, 3.05, -62.40]])
sub03_fid = np.array([[-74.59, -5.49, -64.53],[-1.54, 90.67, -53.76],[80.37, -4.51, -66.42]])
sub04_fid = np.array([[-76.96, 1.07, -71.03],[1.50, 88.75, -53.75],[81.86, -4.60, -68.93]])
sub05_fid = np.array([[-72.97, -1.70, -65.11],[1.63, 92.95, -51.98],[76.08, -4.46, -64.24]])
sub06_fid = np.array([[-76.17, -5.5, -60.15],[-0.56, 89.51, -41.6],[79.06, -8.84, -64.98]])
sub07_fid = np.array([[-73.79, -6.67, -63.82],[2.51, 87.14, -53.10],[78.87, -3.93, -57.46]])
sub08_fid = np.array([[-78.68, 5.50, -65.51],[3.49, 90.70, -53.68],[80.82, -0.45, -60.79]])
sub09_fid = np.array([[-71.48, 1.57, -71.50],[2.47, 88.66, -50.76],[73.32, 3.54, -71.47]])
sub10_fid = np.array([[-80.44, -8.52, -50.56],[-3.37, 89.98, -32.02],[84.89, 5.83, -44.76]])
sub11_fid = np.array([[-74.63, 5.59, -63.42],[3.12, 87.01, -47.96],[77.48, 0.51, -65.50]])
sub12_fid = np.array([[-77.75, -1.49, -70.57],[1.48, 92.32, -52.38],[84.13, 7.39, -62.08]])

for subject in ['sub01','sub02','sub03','sub04','sub05','sub06',
                'sub07','sub08','sub09','sub10','sub11','sub12']:
    for sess in range(60):
        sess = str(sess+1)
        print(sess)
        info = read_info(ROOT + 'meg_pku_raw/'+subject+'/run'+sess+'_tsss.fif')
        plot_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                           surfaces="head-dense", dig=True, eeg=[],
                           meg='sensors', show_axes=True,
                           coord_frame='meg')
        view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                           focalpoint=(0., 0., 0.))

        fiducials = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
        print(fiducials)
        sub_fid = 'sub'+subject[3:]+'_fid'
        for i in range(3):
            fiducials[i]['r'] = locals()[sub_fid][i, :]*0.001
        print(fiducials)

        coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

        coreg.fit_fiducials(verbose=True)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

        coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

        coreg.omit_head_shape_points(distance=5. / 1000)  # distance is in meters

        coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
#mne.viz.set_3d_view(fig, **view_kwargs)

        dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
        print(
            f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
            f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
        )
#input('Press enter to continue')
        mne.write_trans(ROOT + 'meg_pku_raw/'+subject+'/run'+sess+'_trans.fif', coreg.trans)
