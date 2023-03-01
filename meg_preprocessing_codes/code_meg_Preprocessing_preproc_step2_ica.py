import mne, os
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#mpl.use('macosx')

ROOT = './'
# delay = 39.5 #ms
#===========================#

for subject in ['sub01','sub02','sub03','sub04','sub05','sub06',
                'sub07','sub08','sub09','sub10','sub11','sub12']:
    for sess in range(1, 61): # #run
        print(subject, sess) 
        if not os.path.isfile(ROOT + subject+'/run'+str(sess)+'_ica_raw.fif'):
            bads = []
            for line in open(ROOT + 'meg_log/'+subject+'/run'+str(sess)+'_log.txt'):
                if line[0:19] == 'Static bad channels':
                    mid = line.strip().split()
                    for m in mid[4:]:
                        if len(m) == 3:
                            m = '0'+m
                        bads.append('MEG'+m)
                    break
            print(bads)

            #---------------------------------- Filtering ----------------------------------- #
            print('importing raw...')
            raw = mne.io.read_raw_fif(ROOT + subject + '/run' + str(sess) + '_tsss.fif', preload=True)
            events = mne.find_events(raw, min_duration=0.002)
            # raw.crop(tmax=[2][0]/1000)
            # events = mne.find_events(raw, min_duration=0.002)

            raw_filt = raw.filter(1, 40, method='iir', skip_by_annotation='bad_acq_skip')
            raw_filt.info['bads'] = bads
            # raw_filt.plot(events=events, event_color='orange', n_channels=70, duration=50)
            # raw_filt.plot_psd(fmax=50)

            # ----------------------------------- Annotation ------------------------------------ #
            # compute and save ICA
            # events[:, 0] = events[:, 0] - raw.first_samp + delay
            '''
                The first_samp attribute of Raw objects is an integer representing the number of time samples that passed between
                 the onset of the hardware acquisition system and the time when data recording started. This approach to sample 
                 numbering is a peculiarity of VectorView MEG systems, but for consistency it is present in all Raw objects 
                 regardless of the source of the data. In other words, first_samp will be 0 in Raw objects loaded from 
                 non-VectorView data files. 
            '''

            print('read annotations')
            badspan_fname = ROOT + subject + '/run' + str(sess) + '_badspan-annot.fif'
            annot_badspan = mne.read_annotations(badspan_fname)
            for ind in range(len(annot_badspan.description)):
                annot_badspan.description[ind] = 'bad'
            raw_filt.set_annotations(annot_badspan, emit_warning=False)
            print(raw_filt.annotations)

            #----------------------------------- Interpolate ------------------------------------ #
            # compute and save ICA
            # events[:, 0] = events[:, 0] - raw.first_samp + delay

            event_id = dict(begin=1, end=2)
            assert events[0][-1] == 1
            assert events[1][-1] == 2

            raw_filt.info['bads'] = bads
            raw_filt.interpolate_bads(reset_bads=True)  # remove bad
            raw_filt.save(ROOT + subject + '/run' + str(sess) + '_raw.fif', overwrite=True)
            # ----------------------------------- ICA ------------------------------------ #
            # compute and save ICA
            # events[:, 0] = events[:, 0] - raw.first_samp + delay
            #
            #
            ica = mne.preprocessing.ICA(n_components=0.95, method='fastica', random_state=23)
            print('fitting ica...')
            picks_meg = mne.pick_types(raw_filt.info, meg=True, eeg=False, eog=False,
                                       stim=False, exclude='bads')
            reject = dict(mag=5e-12, grad=4000e-13)
            decim = 3
            ica.fit(raw_filt, picks=picks_meg, decim=decim, reject=reject, reject_by_annotation=True)

            ica.plot_sources(raw_filt)
            ica.plot_components(picks=range(ica.n_components_), inst=raw_filt, outlines='skirt')
            input('Press enter to continue')

            # ica.plot_overlay(raw_filt)
            # ica.plot_properties(raw_filt, picks=ica.exclude)
            print(ica.exclude)
            #
            print('Saving...')
            ica.save(ROOT + subject + '/run' + str(sess) + '_ica_parameter.fif', overwrite=True)
            raw_filt_ica = ica.apply(raw_filt, exclude=ica.exclude)
            # raw_filt_ica.plot_psd(fmax=50)
            raw_filt_ica.save(ROOT + subject+'/run'+str(sess)+'_ica_raw.fif')
            del raw, ica

