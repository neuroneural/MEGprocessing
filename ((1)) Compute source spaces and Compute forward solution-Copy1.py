#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne

subject = 'sub-01'
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects'

src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
print(src)


# In[2]:


import os

# set the SUBJECTS_DIR environment variable
subjects_dir = '/data/users2/mjafarlou1/freesurfer/subjects'
os.environ['SUBJECTS_DIR'] = subjects_dir

subject = 'sub-01'
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)

plot_bem_kwargs = dict(
    brain_surfaces='white',  # Specify the color of the brain surfaces
    src=src,  # Provide the source space
    show=True  # Display the plot
)

mne.viz.plot_bem(subject=subject, **plot_bem_kwargs)


# In[ ]:


sphere = (0.0, 0.0, 0.04, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)  # just for speed!
print(vol_src)

# remove the duplicate argument for 'src' from plot_bem_kwargs
plot_bem_kwargs.pop('src', None)

# Make sure to provide the 'subject' argument when calling plot_bem()
mne.viz.plot_bem(subject=subject, src=vol_src, **plot_bem_kwargs)


# In[ ]:


from pathlib import Path

subjects_dir = Path(subjects_dir)
subject = Path(subject)

surface = subjects_dir / subject / "bem" / "inner_skull.surf"
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)  # Just for speed!
print(vol_src)

# Remove the duplicate argument for 'src' from plot_bem_kwargs
plot_bem_kwargs.pop('src', None)

# Make sure to provide the 'subject' argument when calling plot_bem()
mne.viz.plot_bem(subject=subject, src=vol_src, **plot_bem_kwargs)


# In[ ]:


#print(vol_src)

# Remove the duplicate argument for 'src' from plot_bem_kwargs
#plot_bem_kwargs.pop('src', None)

fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=vol_src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)


# In[ ]:


###Compute forward solution


conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject="sub-01", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)


# In[ ]:


#run for just one record

raw_fname = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg/sub-01_task-RDR_run-37_meg.fif"

#trans_fname = "/data/users2/mjafarlou1/freesurfer/subjects/meg_coreg_raw/sub-01/sub-03_task-RDR_run-2_trans.fif"
trans_fname = "/data/users2/mjafarlou1/freesurfer/subjects/sub-01/coreg/sub-01_task-RDR_run-37_trans.fif"

trans = mne.read_trans(trans_fname)

fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)
print(fwd)


# In[ ]:


raw = mne.io.read_raw_fif(raw_fname, preload=True)
num_raw_channels = len(raw.info['ch_names'])
print("Number of channels in raw data:", num_raw_channels)



# In[ ]:


num_forward_channels = len(fwd['info']['ch_names'])
print("Number of channels in the forward solution:", num_forward_channels)



# In[ ]:


raw = mne.io.read_raw_fif(raw_fname, preload=True)
forward_channels = fwd['info']['ch_names']

raw_channels = raw.info['ch_names']
missing_channels = set(raw_channels) - set(forward_channels)

print("Missing channels in the forward solution:", missing_channels)


# In[ ]:


import os
import mne

# Directory paths
raw_directory = "/data/users2/nshor/Multiband_with_MEG/sub-01/meg"
trans_directory = "/data/users2/mjafarlou1/results/coregistration/sub-01"
output_directory = "/data/users2/mjafarlou1/results/forwardsolution/sub-01" 
    
    # Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of all .fif files in the raw directory
raw_files = [file for file in os.listdir(raw_directory) if file.endswith(".fif")]

# Iterate over each raw file
for raw_file in raw_files:
    # Construct the full file paths
    raw_file_path = os.path.join(raw_directory, raw_file)
    trans_file_path = os.path.join(trans_directory, os.path.splitext(raw_file)[0] + "-trans.fif")

    # Read the transformation matrix
    trans = mne.read_trans(trans_file_path)

    # Make the forward solution
    fwd = mne.make_forward_solution(
        raw_file_path,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=None,
        verbose=True,
    )

    # Print the forward solution
    print(fwd)
    

    # Save the forward solution as .fif file
    output_file_path = os.path.join(output_directory, os.path.splitext(raw_file)[0] + "_fwd.fif")
    mne.write_forward_solution(output_file_path, fwd, overwrite=True)


# In[ ]:





# In[ ]:





# In[ ]:


print(f"Before: {src}")
print(f'After:  {fwd["src"]}')


# In[ ]:





# In[ ]:


leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)


# In[ ]:


fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)


# In[ ]:





# In[ ]:




