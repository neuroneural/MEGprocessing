import mne, os
import subprocess
import os.path as op

spacing = 'oct6'
bem_ico = 5
ROOT = './'
subjects_dir = op.join(ROOT, 'mri')
mri_dir = op.join(ROOT, 'mri')

for subject in ['sub01','sub02','sub03','sub04','sub05','sub06',
                'sub07','sub08','sub09','sub10','sub11','sub12']:

    # make BEMs using watershed bem
    # NOTE: Use MNE version >= 20 or set overwrite=True!
    bem_surf_fname = op.join(subjects_dir, subject, 'bem',
                             f'{subject}-ico{bem_ico}-bem.fif')
    bem_sol_fname = op.join(subjects_dir, subject, 'bem',
                            f'{subject}-ico{bem_ico}-bem-sol.fif')
    src_fname = op.join(subjects_dir, subject, 'bem',
                        f'{subject}-ico{bem_ico}-src.fif')

    mne.bem.make_watershed_bem(subject,
                               subjects_dir=subjects_dir,
                               show=False,
                               verbose=False,
                               overwrite=True)
    # make BEM models
    # ico5 is for downsamping
    bem_surf = mne.make_bem_model(
        subject,
        ico=bem_ico,
        conductivity=[0.3],  # for MEG data, 1 layer model is enough
        subjects_dir=subjects_dir)
    mne.write_bem_surfaces(bem_surf_fname, bem_surf)
    # make BEM solution
    bem_sol = mne.make_bem_solution(bem_surf)
    mne.write_bem_solution(bem_sol_fname, bem_sol)

    # Create the surface source space
    src = mne.setup_source_space(subject, spacing, subjects_dir=subjects_dir)
    mne.write_source_spaces(src_fname, src, overwrite=True)

    #bug fixed: https://mne.discourse.group/t/issues-with-make-scalp-surfaces/4016
    def run_command(command, log_file):
        with open(log_file, "wb") as f:
            proc = subprocess.Popen(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            for line in proc.stdout:
                f.write(line)
        if proc.wait() != 0:
            raise RuntimeError("command failed")

    def process_subject_head(subject):
        subject_mri_dir = op.join(mri_dir, subject)
        # When encounter topology errors, use "--force" option.
        # When things go wrong, check "{subject_id}_make_scalp_surfaces.txt" inside
        # the subject's MRI directory to find out why.
        # When practice with defaced data, use "--no-decimate" option.
        run_command([
            "mne", "make_scalp_surfaces", "-s", subject, "-d", subjects_dir,
            "--force", "--verbose"
        ], op.join(subject_mri_dir, f"{subject}_make_scalp_surfaces.txt"))
        print(f"Created high-resolution head surfaces for {subject}")
    

    process_subject_head(subject)

#mne coreg
#export SUBJECTS_DIR=///mri
