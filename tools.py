# -*- coding: utf-8 -*-
"""
Useful functions for the NMF localization project

@author: F.Soldevila (Github: @cbasedlf)
"""
#%% Import stuff
import numpy as np
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
#plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Fourier transforms & filtering in Fourier Domain

def ft2(g,delta):
    '''
    ft2 performs a discretized version of a Fourier Transform by using DFT

    Parameters
    ----------
    g : input field (sampled discretely) on the spatial domain
    delta : grid spacing spatial domain (length units)

    Returns
    -------
    G : Fourier Transform

    '''
    G = fftshift(fft2(ifftshift(g)))*delta**2
    return G

def ift2(G,delta_f):
    '''
    ift2 performs a discretized version of an Inverse Fourier Transform
    by using DFT

    Parameters
    ----------
    G : input field (sampled discretely) on the frequency domain
    delta_f : grid spacing frequency domain (1/length units)

    Returns
    -------
    g : Inverse Fourier Transform

    '''
    n = G.shape[0]
    g = ifftshift(ifft2(fftshift(G)))*(n*delta_f)**2
    return g

def filt_fourier(img,filt_func):
    '''
    filt_fourier filters an image in the Fourier domain.
    To do so, it uses [filt_func]. It multiplies that mask to the 
    Fourier transform of the input image [img], thus eliminating some 
    frequency content. Then it goes back to image domain.

    Parameters
    ----------
    img : Input image (to be filtered)
    filt_func : Filtering mask in the Fourier domain
    
    Returns
    -------
    img_filt : Filtered image

    '''
    # Go to Fourier domain
    img_k = fftshift(fft2(fftshift(img)))
    # Apply filter
    img_k_filt = img_k*filt_func
    # Go back to image domain
    img_filt = np.abs(ifftshift(ifft2(ifftshift(img_k_filt))))
    return img_filt

def buildGauss(px,sigma,center,phi):
    """
    buildGauss generates a Gaussian function in 2D. Formula from
    https://en.wikipedia.org/wiki/Gaussian_function

    Parameters
    ----------
    px : image size of the output (in pixels)
    sigma : 2-element vector, sigma_x 
    and sigma_y for the 2D Gaussian
    center : 2-element vector, center position
    of the Gaussian in the image
    phi : Rotation angle for the Gaussian

    Returns
    -------
    gaus : 2D image with the Gaussian

    """
    #Generate mesh
    x = np.linspace(1,px,px)
    X,Y = np.meshgrid(x,x)
    
    #Generate gaussian parameters
    a = np.cos(phi)**2/(2*sigma[0]**2) + np.sin(phi)**2/(2*sigma[1]**2)
    b = -np.sin(2*phi)/(4*sigma[0]**2) + np.sin(2*phi)/(4*sigma[1]**2)
    c = np.sin(phi)**2/(2*sigma[0]**2) + np.cos(phi)**2/(2*sigma[1]**2)
    
    #Generate Gaussian
    gaus = np.exp(-(a*(X-center[0])**2 + 2*b*(X-center[0])*(Y-center[1]) + c*(Y-center[1])**2))
    
    return gaus

#%% Image manipulation

def cropROI(img,size,center_pos):
    '''
    cropROI gets a ROI with a desired size, center at a fixed position

    Parameters
    ----------
    img : input image
    size : size of the ROI (2 element vector, size in [rows,cols] format)
    center_pos : central position of the ROI

    Returns
    -------
    cropIMG = cropped ROI of the image

    '''
    if img.shape[0]< size[0] or img.shape[1] < size[1]:
        print('Size is bigger than the image size')
        return img
    else:
        center_row = center_pos[0]
        center_col = center_pos[1]
        semiROIrows = int(size[0]/2)
        semiROIcols = int(size[1]/2)
        cropIMG = img[center_row - semiROIrows : center_row + semiROIrows,
                      center_col - semiROIcols : center_col + semiROIcols]
        pass
    
    return cropIMG

def show_img(img, fig_size = False, colormap = 'viridis'):
    '''
    show_img plots a single matrix as an image

    Parameters
    ----------
    img : matrix to plot
    fig_size: size of the figure (inches)
    colormap: colormap of the plot

    '''
    if fig_size == False:
        fig_size = (5,5)
        pass
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=fig_size)
    im1 = ax.imshow(img, interpolation = "nearest", cmap = colormap)    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im1, cax = cax, ax = ax)
    ax.set_aspect(1)
    return ax

def show_Nimg(hypercube,fig_size=False,colormap='viridis'):
    '''
    show_Nimg generates a grid plot from a set of 2D images.

    Parameters
    ----------
    hypercube : Set of 2D images. Third axis should be the image number
    fig_size : size of the figure 
    colormap : TYPE, optional
        DESCRIPTION. The default is 'viridis'.

    '''
    if fig_size == False:
        fig_size = (8,8)
        pass
    Nimg = hypercube.shape[2]
    nrows = int(np.ceil(np.sqrt(Nimg)))
    fig, ax = plt.subplots(nrows = nrows, ncols = nrows, figsize = fig_size)
    counter = 0
    for rowidx in range(0,nrows):
        for colidx in range(0,nrows):
            if counter < Nimg:
                im = ax[rowidx,colidx].imshow(hypercube[:,:,counter],
                                              cmap = colormap)
                ax[rowidx,colidx].set_aspect(1)
                divider = make_axes_locatable(ax[rowidx,colidx])
                cax = divider.append_axes('right', size='5%', pad = 0.1)
                fig.colorbar(im, cax = cax, ax = ax[rowidx,colidx])
                counter += 1
                pass
            pass
        pass
    plt.tight_layout()
    plt.show()
    pass

def show_vid(hypercube, rate, fig_size = False, colormap='viridis',
             cbarfix = False, loop = False):
    '''
    show_vid creates an animation showing the frames of a video.
    The input is a 3D array, where the third dimension corresponds to time

    Parameters
    ----------
    hypercube : input array
    rate : frame rate (in ms)
    fig_size : size of the plot
    colormap : colormap
    cbarfix : option to have the same colorbar range for all frames (True)
                or not (False)

    '''
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if fig_size == False:
        fig_size = (6,6)
        pass
    
    fig , ax = plt.subplots()
    
    if cbarfix == True:
        cmin = np.min(hypercube)
        cmax = np.max(hypercube)
        cbarlimits = np.linspace(cmin,cmax,10,endpoint=True)
    pass

    def plot_img(i):
        plt.clf()
        plt.suptitle('Frame #' + str(i))        
        if cbarfix == True:
            im1 = plt.imshow(hypercube[:,:,i],vmin = cmin, vmax = cmax,
                             cmap = colormap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.6)
            ax.set_aspect(1)
            plt.colorbar(im1,cax = cax, ax = ax, ticks = cbarlimits)
        else:
            im1 = plt.imshow(hypercube[:,:,i], cmap = colormap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.6)
            ax.set_aspect(1)
            plt.colorbar(im1,cax = cax, ax = ax)
            pass
        plt.show()
        pass
    anim = animation.FuncAnimation(fig, plot_img, frames = hypercube.shape[2],
                                   interval = rate, repeat = loop)
    return anim

#%% General stuff
def iseven(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is even/odd

    '''    
    return number % 2 == 0

def isodd(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is odd/even

    '''    
    return number % 2 != 0
