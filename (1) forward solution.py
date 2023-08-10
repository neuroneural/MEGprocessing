#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne

subject = 'sub-01'
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects'

src = mne.setup_source_space(subject, spacing='all', n_jobs=-1, add_dist='patch', subjects_dir=subjects_dir)
print(src)


# In[2]:


import pylab as plt
pp = src[0]['rr']
plt.plot(pp[:,0], pp[:,1], '.', ms=0.05)
plt.show()


# In[3]:


import os

# set the SUBJECTS_DIR environment variable
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects'
os.environ['SUBJECTS_DIR'] = subjects_dir

subject = 'sub-01'
src = mne.setup_source_space(subject, spacing='all', n_jobs=-1, add_dist='patch', subjects_dir=subjects_dir)

plot_bem_kwargs = dict(
    brain_surfaces='white',  # Specify the color of the brain surfaces
    src=src,  # Provide the source space
    show=True  # Display the plot
)

mne.viz.plot_bem(subject=subject, **plot_bem_kwargs)


# In[4]:


###Compute forward solution


conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject="sub-01", ico=None, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)


# In[5]:


#run for just one record

raw_fname = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg/sub-01_task-RDR_run-37_meg.fif"

#trans_fname = "/data/users2/mjafarlou1/freesurfer/subjects/meg_coreg_raw/sub-01/sub-03_task-RDR_run-2_trans.fif"
trans_fname = "/data/users2/mjafarlou1/results/coregistration/sub-01/sub-01_task-RDR_run-37_meg-trans.fif"

trans = mne.read_trans(trans_fname)

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


# In[6]:


# convert the forward model with free source orientation to a model 
# with orientation fixed to the direction orthogonal to each vertex 
# on the surface
fwd = mne.convert_forward_solution(fwd, surf_ori=True, 
                                   force_fixed=True, copy=False, 
                                   use_cps=True,
                                   verbose=None)


# In[7]:


raw = mne.io.read_raw_fif(raw_fname, preload=True)
num_raw_channels = len(raw.info['ch_names'])
print("Number of channels in raw data:", num_raw_channels)



# In[8]:


num_forward_channels = len(fwd['info']['ch_names'])
print("Number of channels in the forward solution:", num_forward_channels)



# In[9]:


raw = mne.io.read_raw_fif(raw_fname, preload=True)
forward_channels = fwd['info']['ch_names']

raw_channels = raw.info['ch_names']
missing_channels = set(raw_channels) - set(forward_channels)

print("Missing channels in the forward solution:", missing_channels)


# In[11]:


# Save the forward solution as .fif file
output_directory = "/data/users2/mjafarlou1/results/forwardsolution/sub-01" 

output_file_path = os.path.join(output_directory, os.path.splitext(raw_file)[0] + "_fwd.fif")
mne.write_forward_solution(output_file_path, fwd, overwrite=True)

