# -*- coding: utf-8 -*-
"""
Import dataset (from Matlab)
Clean it a little bit (do binning, filter background, etc)
Perform NMF on the video
Recover spatial locations and show

@author: F.Soldevila (Github: @cbasedlf)
"""

#%% Import libraries, etc.
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import scipy.signal
from skimage import restoration
from PIL import Image
from sklearn.decomposition import NMF
import tools as ts

#%% Load dataset 
f = h5py.File('./data/brain_slice.mat', mode = 'r')
video = np.array(f['video_data'])
video = np.moveaxis(video,0,2) #rearrange to px*px*time form

#%% Define experimental variables
#Crop central part of the images (define size of the frame)
frame_size = np.asarray((890,940), dtype = int)
#Number of emitters in the sample
numbeads = 11
#%% Clean dataset: binning (to go faster) + filtering

#check orientation of the frame (vertical video or horizontal)
if frame_size[0] > frame_size[1]:
    ORIENTATION = 'vertical'
elif frame_size[0] < frame_size[1]:
    ORIENTATION = 'horizontal'
else:
    ORIENTATION = 'square'

#preallocate cropped video
video_small = np.zeros((frame_size[0], frame_size[1], video.shape[2]),
                       dtype = 'uint16')
#crop the video
for idx in range(video.shape[2]):
    video_small[:,:,idx] = ts.cropROI(video[:,:,idx], size = frame_size,
                                        center_pos = (int(frame_size[0]/2),
                                                    int(frame_size[1]/2)))
#Do binning
BINNING = 2 #bin size
#calculate new frame sizes after binning
new_frame_size = (frame_size/BINNING).astype(int)
short_side = np.min(new_frame_size)#calculate short side
long_side = np.max(new_frame_size)#calculate long side
#preallocate binned video
video_binned = np.zeros((new_frame_size[0], new_frame_size[1], video.shape[2]))
#do the binning
print('Binning...')
for idx in range(video.shape[2]):
    temp = video_small[:,:,idx]
    temp = Image.fromarray(temp).resize((new_frame_size[1], new_frame_size[0]),
                                        resample = Image.NEAREST)
    temp = np.array(temp)
    video_binned[:,:,idx] = temp
print('done')

print('Filtering...')
#Filter low frequency background (envelope)
video_highpass = np.zeros(video_binned.shape) #preallocate
#Build gaussian filter. Sizes have to be manually tuned for now
gaussian = ts.buildGauss(short_side, (3,3),
                           (int(short_side/2), int(short_side/2)), 0)

#pad the filter so it gets same size as the frame
#(gaussian is always a square image, but video might be rectangular)
if ORIENTATION == 'vertical':
    padsize = long_side - short_side
    #take care of padsize, otherwise filter might be smaller than the image
    if ts.iseven(padsize):
        gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2)), (0,0)))
    else:
        gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2) + 1), (0,0)))
elif ORIENTATION == 'horizontal':
    padsize = long_side - short_side
    #take care of padsize, otherwise filter might be smaller than the image
    if ts.iseven(padsize):
        gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2))))
    else:
        gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2) + 1)))
else: #if is neither hor. nor vert., is square and no need to do anything
    pass
lowfreqfilter = 1 - gaussian

#do the filtering for each frame
for idx in range(0,video.shape[2]):
    video_highpass[:,:,idx] = ts.filt_fourier(video_binned[:,:,idx],
                                                lowfreqfilter)
print('done')
# ts.show_vid(video_highpass, rate = 200) #show filtered video

#%%Do NMF on the pre-processed video (croped,binned, and filtered)
print('Doing NMF...')
numframes = video.shape[2] #get number of frames of the video
#set rank as number of beads + background noise
#(not strictly needed to add the noise term, but it helps the NMF unmixing)
rank = numbeads + 1
#reshape intro matrix form
X = np.reshape(video_highpass, (new_frame_size[0] * new_frame_size[1], numframes))
#Do the NMF without a priori knowledge in the initialization init='nndsvd'
#Create the model:
model = NMF(n_components = rank, init = 'nndsvd',
            random_state = 0, max_iter = 3000, solver = 'cd', l1_ratio = 0.5,
            beta_loss = 2, verbose = 0, alpha_W = 1.5, alpha_H = 0.5)
#Run the model, store spatial fingerprints in W:
W = model.fit_transform(X)
#Store temporal activities in H:
H = model.components_
print('done')
#Reshape spatial fingerprints in 3D tensor form:
fingerprints_raw = np.reshape(W, (new_frame_size[0], new_frame_size[1], rank))
#show recovered fingerprints
ts.show_Nimg(fingerprints_raw) 
#identify the fingerprint which corresponds to noise (should be the one with
#the lowest maximum intensity value). Remove it from the group before localization
max_intensities = np.max(fingerprints_raw, axis = (0,1))
noise_fingerprint_index = np.where(max_intensities == np.min(max_intensities))[0]
fingerprints = np.delete(fingerprints_raw,noise_fingerprint_index,axis = 2) #remove noise fingerprint
ts.show_Nimg(fingerprints) #show

#%% Do DECONVOLUTIONS with recovered fingerprints PAIRWISE
print('Doing localization...')
obj_img = []
for outidx in range(numbeads):

    deconv = [] #initialization
    model_fingerprint = fingerprints[:,:,outidx] #model fingerprint
    for inidx in range(numbeads):
        temp = fingerprints[:,:,inidx] #fingerprint to deconvolve
        #Do deconvolution
        temp_deconv = restoration.wiener(model_fingerprint, temp, balance = 1e7)
        deconv.append(temp_deconv)#store deconvolution
    deconv = np.moveaxis(np.asarray(deconv), 0, 2)

    #Build gaussian filter and clean the deconvolutions
    #calculate correlation size
    deconv_size = np.asarray(deconv.shape[0:2]).astype(int)
    short_deconvside = np.min(deconv_size)
    long_deconvside = np.max(deconv_size)
    #Build gaussian filter
    gaussian = ts.buildGauss(short_deconvside, (35,35),
                        (int(short_deconvside/2), int(short_deconvside/2)), 0)
    #pad the filter so it gets same size as the frame
    if ORIENTATION == 'vertical':
        padsize = long_deconvside - short_deconvside
        if ts.iseven(padsize):
            gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2)), (0,0)))
        else:
            gaussian = np.pad(gaussian,
                              ((int(padsize/2), int(padsize/2) + 1), (0,0)))
    elif ORIENTATION == 'horizontal':
        padsize = long_deconvside - short_deconvside
        if ts.iseven(padsize):
            gaussian = np.pad(gaussian,
                              ((0,0), (int(padsize/2), int(padsize/2))))
        else:
            gaussian = np.pad(gaussian,
                              ((0,0), (int(padsize/2), int(padsize/2) + 1)))
    else:
        pass
    lowfreqfilter = 1 - gaussian
    deconv_hp = np.zeros((deconv_size[0], deconv_size[1], rank)) #preallocate
    #Set number of pixels at the borders to set as zero
    #(helps removing artifacts from deconvolution)
    FRAME_CROP_SIZE = 5
    for idx in range(numbeads):
        deconv_hp[:,:,idx] = ts.filt_fourier(deconv[:,:,idx], gaussian)
        #do thresholding (clean deconvolutions to improve results)
        deconv_hp[:,:,idx] *= (deconv_hp[:,:,idx] > 0.85 * np.max(
                                                        deconv_hp[:,:,idx]))
        #set borders of the image to zero (remove artifacts)
        deconv_hp[0:FRAME_CROP_SIZE,:,idx] = 0
        deconv_hp[-FRAME_CROP_SIZE::,:,idx] = 0
        deconv_hp[:,0:FRAME_CROP_SIZE,idx] = 0
        deconv_hp[:,-FRAME_CROP_SIZE::,idx] = 0
    #add all individual images
    obj_img.append(np.sum(deconv_hp, 2))
obj_img = np.moveaxis(np.asarray(obj_img), 0, 2)

###Cross-correlate and shift each image, then add all together
#Preallocation of the position of the peaks
peak_subimage = np.zeros((numbeads, numbeads, 2))
#Preallocation of the shifts of the peaks with respect to center of the image
peak_shifts = np.zeros_like(peak_subimage)
for i in range(numbeads):
    model_subimage = obj_img[:,:,i] / (obj_img[:,:,i] + 1e-18)
    deconv_subimage = []
    for j in range(numbeads):
        temp = obj_img[:,:,j] / (obj_img[:,:,j] + 1e-18)
        deconv_subimage.append(scipy.signal.correlate(model_subimage, temp,
                                                      mode = 'full'))
        deconv_subimage[j] = (deconv_subimage[j] - np.min(
            deconv_subimage[j])) / (np.max(deconv_subimage[j]) - np.min(
                deconv_subimage[j]) + 1e-18)
        temp1, temp2 =  np.where(deconv_subimage[j] == np.max(
                                                        deconv_subimage[j]))
        peak_subimage[i,j,0] = temp1[0]
        peak_subimage[i,j,1] = temp2[0]
        peak_shifts[i,j,:] = peak_subimage[i,j,:] - peak_subimage[0,0,0]
    deconv_subimage = np.moveaxis(np.asarray(deconv_subimage), 0, 2)

subimg_shifted = []
for i in range(numbeads):
    temp = np.zeros_like(obj_img[:,:,0])
    for j in range(numbeads):
        temp += np.roll(obj_img[:,:,j], (int(peak_shifts[i,j,0]),
                                        int(peak_shifts[i,j,1])), axis = (0,1))
    subimg_shifted.append(temp)
subimg_shifted = np.moveaxis(np.asarray(subimg_shifted),0,2)

subimg_shifted_corrected = []
for i in range(numbeads):
    subimg_shifted_corrected.append(np.roll(subimg_shifted[:,:,i],
                                    (int(peak_shifts[0,i,0]),
                                     int(peak_shifts[0,i,1])), axis = (0,1)))
subimg_shifted_corrected = np.moveaxis(np.asarray(subimg_shifted_corrected), 0, 2)

# Set global shift to correct image warping (just to ~center on the FoV)
global_shift = [0,-50]
#Show localization recovery
obj = np.sum(np.roll(subimg_shifted_corrected,global_shift, [0,1]), 2)
ts.show_img(obj)
print('done')