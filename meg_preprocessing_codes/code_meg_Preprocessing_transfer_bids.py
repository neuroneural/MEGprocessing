import matplotlib
import pathlib

import mne
import mne_bids

matplotlib.use('Qt5Agg')

Root = './'

subs = ['01','02','03','04','05','06','07','08','09','10','11','12']

for ss in subs:
    for rr in range(1, 61):
        rr = str(rr)
        raw = mne.io.read_raw(Root + 'meg_pku_raw/sub' + ss + '/run' + rr + '_ica_raw.fif')

        trans = mne.read_trans(Root + 'meg_pku_raw/sub' + ss + '/run' + rr + '_trans.fif')
        print(trans)

        events = mne.find_events(raw, shortest_event=1)
        print(events.shape)
        events = events[0:2, :]
        print(events.shape)

        event_id = {
            'Beg': 1,
            'End': 2,
            'ans': 3,
        }

        raw.info['line_freq'] = 60

        out_path = pathlib.Path(Root + 'meg_pku_BIDS_format_clean/')

        bids_path = mne_bids.BIDSPath(subject=ss,
                                      task='language',
                                      run=rr,
                                      root=out_path)

        mne_bids.write_raw_bids(raw, bids_path=bids_path, events_data=events,
                                event_id=event_id, overwrite=True)


        t1w_bids_path = mne_bids.BIDSPath(subject=ss,
                                          root=out_path,
                                          suffix='T1w')
        t1_fname = Root + 'meg_pku/T1w/sub' + ss + '_T1w.nii.gz'

        landmarks = mne_bids.get_anat_landmarks(
            t1_fname,  # path to the MRI scan
            info=raw.info,  # the MEG data file info from the same subject as the MRI
            trans=trans,  # our transformation matrix
            fs_subject='sub'+ss,  # FreeSurfer subject
            fs_subjects_dir=Root+'meg_pku/mri',  # FreeSurfer subjects directory
        )

        # We use the write_anat function
        t1w_bids_path = mne_bids.write_anat(
            image=t1_fname,  # path to the MRI scan
            bids_path=t1w_bids_path,
            landmarks=landmarks,  # the landmarks in MRI voxel space
            verbose=True,  # this will print out the sidecar file
            overwrite=True
        )
        anat_dir = t1w_bids_path.directory

mne_bids.print_dir_tree(out_path)
