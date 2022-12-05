'''
Rank estimation from a dataset.
Perform NMF with different ranks, calculate data fidelity.
Plot results
'''
#%% Import libraries, etc.
import sys
sys.path.append('../NMF_lightfield/localization-shackhartmann')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from PIL import Image
from sklearn.decomposition import NMF
import tools_fer as tfer
#%% Load dataset & ground truth activations
#load ground truth
fgt = sio.loadmat('./10052022/data_10052022_001_gt.mat')
act_gt = fgt['pat'].T
#normalize ground truth traces (for doing comparisons later)
act_gt_norm = tfer.norm_dimension(act_gt, dim = 1, mode='0to1')
#load dataset
f = h5py.File('./10052022/data_10052022_001.mat', mode = 'r')
video = np.array(f['video_data'])
video = np.moveaxis(video,0,2) #rearrange to px*px*time form

#%% Clean dataset: binning (to go faster) + filtering

#Crop central part of the images (sizes should be 2 x even_number)
frame_size = np.asarray((880,940), dtype = int)
#check orientation of the frame (vertical video or horizontal)
if frame_size[0] > frame_size[1]:
    ORIENTATION = 'vertical'
elif frame_size[0] < frame_size[1]:
    ORIENTATION = 'horizontal'
else:
    ORIENTATION = 'square'

#preallocate cropped video
video_small = np.zeros((frame_size[0], frame_size[1], video.shape[2]))
#crop the video
for idx in range(video.shape[2]):
    video_small[:,:,idx] = tfer.cropROI(video[:,:,idx], size = frame_size,
                                        center_pos = (int(frame_size[0]/2),
                                                    int(frame_size[1]/2)))
#Do binning
BINNING = 4 #bin size
#calculate new frame sizes after binning
new_frame_size = (frame_size/BINNING).astype(int)
short_side = np.min(new_frame_size)#calculate short side
long_side = np.max(new_frame_size)#calculate long side
#preallocate binned video
video_binned = np.zeros((new_frame_size[0], new_frame_size[1], video.shape[2]))
#do the binning
for idx in range(video.shape[2]):
    temp = video_small[:,:,idx]
    temp = Image.fromarray(temp).resize((new_frame_size[1], new_frame_size[0]),
                                        resample = Image.NEAREST)
    temp = np.array(temp)
    video_binned[:,:,idx] = temp

#Filter low frequency background (envelope)
video_highpass = np.zeros(video_binned.shape) #preallocate
#Build gaussian filter. Sizes have to be manually tuned for now
gaussian = tfer.buildGauss(short_side, (3,3),
                           (int(short_side/2), int(short_side/2)), 0)
#pad the filter so it gets same size as the frame
#(gaussian is always a square image, but video might be rectangular)
if ORIENTATION == 'vertical':
    padsize = long_side - short_side
    #take care of padsize, otherwise filter might be smaller than the image
    if tfer.iseven(padsize):
        gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2)), (0,0)))
    else:
        gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2) + 1), (0,0)))
elif ORIENTATION == 'horizontal':
    padsize = long_side - short_side
    #take care of padsize, otherwise filter might be smaller than the image
    if tfer.iseven(padsize):
        gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2))))
    else:
        gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2) + 1)))
else: #if is neither hor. nor vert., is square and no need to do anything
    pass
lowfreqfilter = 1 - gaussian
#do the filtering
for idx in range(0,video.shape[2]):
    video_highpass[:,:,idx] = tfer.filt_fourier(video_binned[:,:,idx],
                                                lowfreqfilter)
    
#%% Do NMF with different ranks, store results
num_NMF = 5 #Number of iterations for each rank
start_rank = 1 #starting rank value
end_rank = 20 #ending rank value

#Prepare video to be used by the NMF algorithm

#Get real number of beads (from ground truth), this is just to crop the video,
#in a real experiment we would not have this info, but also these frames would
#not be acquired!!!
numbeads = act_gt.shape[1]
#remove ground truth frames (initial part of the acquired video in the lab)
X = video_highpass[:,:,numbeads::].copy()
#reshape intro matrix form
X = np.reshape(X, (new_frame_size[0] * new_frame_size[1], X.shape[2]))

fidelity = np.zeros((end_rank - start_rank,num_NMF)) #initialization
print('Starting rank estimation...')
for rankidx in range(end_rank - start_rank):
    for iteridx in range(num_NMF):
        #Create the model:
        model = NMF(n_components = rankidx + start_rank, init = 'random',
                    max_iter = 2000, solver = 'cd', l1_ratio = 0.5,
                    beta_loss = 2, verbose = 0, alpha_W = 1.5, alpha_H = 0.5)
        #Run the model, store fingerprints in W:
        W = model.fit_transform(X)
        #Store temporal activities in H:
        H = model.components_
        #Calculate loss (Frobenius norm):
        fidelity[rankidx,iteridx] = np.linalg.norm(X - W @ H, 'fro') / X.size
    print('rank = ' + str(rankidx + start_rank) + '... done')

#%% Calculate averages, show results
fidelity_average = np.mean(fidelity, axis = 1)#Calculate mean
fidelity_std = np.std(fidelity, axis = 1)#Calculate standard deviation

#Plot results
fig , ax = plt.subplots(figsize = (10,5))
plt.rcParams['text.usetex'] = True #LaTeX rendering
plt.title('Fidelity values vs NMF rank')
ax.plot(np.arange(start = start_rank, stop = end_rank, step = 1),
        fidelity_average, label = 'mean (' + str(num_NMF) + ' realizations)',
        marker = '1', markersize = 10)
ax.fill_between(np.arange(start = start_rank, stop = end_rank, step = 1),
                fidelity_average - fidelity_std,
                fidelity_average + fidelity_std,
                color = 'blue', alpha = .2, label = 'std')
ax.plot(np.arange(start = start_rank, stop = end_rank, step = 1),
        np.min(fidelity,1), color = 'red', linewidth = 0.8, label = 'min/max')
ax.plot(np.arange(start = start_rank, stop = end_rank, step = 1),
        np.max(fidelity,1), color = 'red', linewidth = 0.8)
ax.set_xlabel('NMF rank')
ax.set_ylabel(r'Loss ($\vert\vert X-WH \vert\vert_{F}$)')
plt.legend()
plt.show()
plt.rcParams['text.usetex'] = False #Turn OFF LaTeX rendering