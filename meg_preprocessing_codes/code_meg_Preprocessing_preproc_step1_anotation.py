import mne, os
import matplotlib as mpl
mpl.use('macosx')

def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)

ROOT = './'
# delay = 39.5 #ms
#===========================#

for subject in ['sub01','sub02','sub03','sub04','sub05','sub06',
                'sub07','sub08','sub09','sub10','sub11','sub12']:
    for sess in range(1, 61): # #run
        print(subject, sess) 
        if not os.path.isfile(ROOT + subject+'/run'+str(sess)+'_badspan-annot.fif'):
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

            raw_filt = raw.filter(1, 40, method='iir')
            raw_filt.info['bads'] = bads
            # raw_filt.plot(events=events, event_color='orange', n_channels=70, duration=50)
            # raw_filt.plot_psd(fmax=50)

            # ----------------------------------- Annotation ------------------------------------ #
            start = (events[1, 0] - raw_filt.first_samp) / 1000
            ans_annot = mne.Annotations(onset=[start],  # in seconds
                                        duration=[1000],  # in seconds, too
                                        description=['ans'])
            raw_filt.set_annotations(ans_annot)
            mne.viz.plot_raw(raw_filt, n_channels = 40)
            plt_show(True)

            print('save annotations')
            badspan_fname = ROOT + subject+'/run'+str(sess)+'_badspan-annot.fif'
            raw_filt.annotations.save(badspan_fname)


