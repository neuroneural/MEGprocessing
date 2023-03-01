import mne, os
import os.path as op

ROOT = './'

#===========================#

for subject in ['sub01','sub02','sub03','sub04','sub05','sub06',
                'sub07','sub08','sub09','sub10','sub11','sub12']:
    for sess in range(1, 61): # #run
        print('**************** sess *********************')

        raw = mne.io.read_raw_fif(ROOT + 'meg_pku_raw/'+subject+'/run'+str(sess)+'_ica_raw.fif', preload=True)
        picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)

        ## generate stcs
        info = raw.info
        trans = mne.read_trans(ROOT+'meg_pku_raw/'+subject+'/run'+str(sess)+'_trans.fif')
        bem = ROOT + 'meg_pku/mri/'+subject+'/bem/'+subject+'-ico5-bem-sol.fif'

        src_fname = ROOT + 'meg_pku/mri/'+subject+'/bem/'+subject+'-ico5-src.fif'
        src = mne.read_source_spaces(fname=src_fname)

        # make forward solution
        fwd_fname = ROOT + 'meg_pku_raw/'+subject+'/run'+str(sess)+'_fwd.fif'
        print('Creating forward solution...')
        if os.path.isfile(fwd_fname):
            print('forward soltion for subj=%s exists, loading file.' % subject)
            fwd = mne.read_forward_solution(fwd_fname)
            print('Done.')
        else:
            print('forward soltion for subj=%s does not exist, creating file.' % subject)
            fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem, meg=True, eeg=False, ignore_ref=True)
            mne.write_forward_solution(fwd_fname, fwd)
            print('Done. File saved.')

        if not op.exists(ROOT + 'meg_pku_raw/'+subject+'/run'+str(sess)+'_cov.fif'):
            cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method=['shrunk', 'diagonal_fixed', 'empirical'], rank='info')
            mne.write_cov(ROOT + 'meg_pku_raw/'+subject+'/run'+str(sess)+'_cov.fif', cov)
        else:
            cov = mne.read_cov(ROOT + 'meg_pku_raw/'+subject+'/run'+str(sess)+'_cov.fif')
        print('-----------------------------------inv--------------')
        # make inverse solution
        fixed = True  # set to False for free estimates
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=None, loose='auto', fixed=False)
        lambda2 = 1.0 / 2.0 ** 2 #SNR=2 for single trail
        print('--------------------------------stcs--------------------')
        stcs = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2, method='dSPM', use_cps=True)
        # stcs = mne.minimum_norm.apply_inverse_raw(raw.crop(0, 10), inv, lambda2, method='dSPM', use_cps=True)

        counter = 1
        if not op.exists(ROOT+'meg_pku_stcs/%s' %subject):
            os.makedirs(ROOT+'meg_pku_stcs/%s' %subject)

        print('------------------------------save-------------------------')
        stcs.save(ROOT + 'meg_pku_stcs/'+subject+'/run'+str(sess)+'.stc')

        del(stcs)
        del(raw)
        del(fwd)
