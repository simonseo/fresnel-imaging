import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift
from scipy.sparse.linalg import eigs, eigsh, LinearOperator, norm as sparsenorm
import matplotlib.pyplot as plt
from scipy.ndimage import edgetaper

def imconv(F, K, output):
    # Convolution with boundary condition
    # General: F_filt = imfilter(F, K, 'full', 'conv', 'replicate')
    # Speedup for small two entry kernels (full)
    if K.shape[0] == 1 and K.shape[1] == 2 and output == 'full':
        F_filt = K[0,1] * F[:, np.concatenate(([0], np.arange(F.shape[1]-1))), :] + K[0,0] * F[:, np.concatenate((np.arange(F.shape[1]-1), [F.shape[1]-1])), :]
    elif K.shape[0] == 2 and K.shape[1] == 1 and output == 'full':
        F_filt = K[1,0] * F[np.concatenate(([0], np.arange(F.shape[0]-1))), :, :] + K[0,0] * F[np.concatenate((np.arange(F.shape[0]-1), [F.shape[0]-1])), :, :]
    else:
        # General model
        F_filt = cv2.filter2D(F, -1, K, borderType=cv2.BORDER_REPLICATE)

    return F_filt


def compute_operator_norm(A, AS, sx):
    """Computes the operator norm for a linear operator AS on images with size sx, 
    which is the square root of the largest eigenvector of AS*A."""
    
    vec_size = sx[0] * sx[1]
    
    # Define the function that returns AS(A(x))
    def ASAfun(x):
        x_img = x.reshape(sx)
        ASAx = AS(A(x_img))
        return ASAx.flatten()
    
    # Compute largest eigenvalue (in this case arnoldi, since scipy implementation 
    # is faster than power iteration)
    sigma = np.random.rand(vec_size)
    lambda_largest, _ = eigsh(LinearOperator((vec_size, vec_size), matvec=ASAfun), 
                              k=1, which='LM', tol=1.e-3, maxiter=10, v0=sigma)
    
    L = np.sqrt(lambda_largest)
    return L


def KSmult(f, ch, db_chs, lambda_cross_ch, lambda_tv):

    # Derivative filters
    dxf = np.array([-1, 1])
    dyf = np.array([-1, 1]).reshape(-1, 1)
    dyyf = np.array([-1, 2, -1]).reshape(-1, 1)
    dxxf = np.array([-1, 2, -1])
    dxyf = np.array([[-1, 1], [1, -1]])
    
    # Result
    KSmultf = np.zeros([f.shape[0], f.shape[1]])

    # Compute tv terms
    i = 0 # Image access term
    if lambda_tv > np.finfo(float).eps:
        i += 1
        fx = convolve2d((lambda_tv * 0.5) * f[:,:,i], dxf, mode='full')
        fx = fx[:, :-1]
        
        i += 1
        fy = convolve2d((lambda_tv * 0.5) * f[:,:,i], dyf, mode='full')
        fy = fy[:-1, :]
        
        sd_w = 0.15 # Weight for second derivatives
        i += 1
        fxx = convolve2d((lambda_tv * sd_w) * f[:,:,i], dxxf, mode='full')
        fxx = fxx[:, :-2]
        
        i += 1
        fyy = convolve2d((lambda_tv * sd_w) * f[:,:,i], dyyf, mode='full')
        fyy = fyy[:-2, :]
        
        i += 1
        fxy = convolve2d((lambda_tv * sd_w) * f[:,:,i], dxyf, mode='full')
        fxy = fxy[:-1, :-1]
        
        # Gather result
        KSmultf = fx + fy + fxx + fyy + fxy

    # Cross-Terms for all adjacent channels
    for adj_ch in range(len(db_chs)):
        
        # Early exit
        if np.sum(lambda_cross_ch) <= np.finfo(float).eps:
            break
        
        # Continue for current channel
        if adj_ch == ch or db_chs[adj_ch]['K'] is None:
            continue
        adjChImg = db_chs[adj_ch]['Image'] # Curr cross channel
        
        # Compute cross terms
        i += 1
        f[:,:,i] = (lambda_cross_ch[adj_ch] * 0.5) * f[:,:,i]
        diag_term = convolve2d(adjChImg, np.fliplr(np.flipud(dxf)), mode='full')
        diag_term = diag_term[:, 1:] * f[:,:,i]
        conv_term = convolve2d(adjChImg * f[:,:,i], dxf, mode='full')
        Sxtf = conv_term[:, :-1] - diag_term
        
        i += 1
        f[:,:,i] = (lambda_cross_ch[adj_ch] * 0.5) * f[:,:,i]
        diag_term = convolve2d(adjChImg, np.fliplr(np.flipud(dyf)), mode='full')
        diag_term = diag_term[1:, :] * f[:,:,i]
        conv_term = convolve2d(adjChImg * f[:,:,i], dyf, mode='full')
        Sytf = conv_term[:-1, :] - diag_term
        
        # Gather result
        KSmultf = KSmultf + Sxtf + Sytf
        
    return KSmultf




def Kmult(f, ch, db_chs, lambda_cross_ch, lambda_tv):
    # Initialize result
    Kmultf = []

    # Compute tv terms
    if lambda_tv > np.finfo(float).eps:
        dxf = np.array([[1, 0, -1]])
        dyf = np.array([[1], [0], [-1]])
        dxxf = np.array([[1, -2, 1]])
        dyyf = np.array([[1], [-2], [1]])
        dxyf = np.array([[1, -1], [-1, 1]])

        fx = convolve2d(f, np.fliplr(np.flipud(dxf)), mode='full')
        fx = (lambda_tv * 0.5) * fx[:, 1:]

        fy = convolve2d(f, np.fliplr(np.flipud(dyf)), mode='full')
        fy = (lambda_tv * 0.5) * fy[1:, :]

        sd_w = 0.15  # weight for second derivatives
        fxx = convolve2d(f, np.fliplr(np.flipud(dxxf)), mode='full')
        fxx = (lambda_tv * sd_w) * fxx[:, 2:]

        fyy = convolve2d(f, np.fliplr(np.flipud(dyyf)), mode='full')
        fyy = (lambda_tv * sd_w) * fyy[2:, :]

        fxy = convolve2d(f, np.fliplr(np.flipud(dxyf)), mode='full')
        fxy = (lambda_tv * sd_w) * fxy[1:, 1:]

        # Gather
        Kmultf = np.concatenate((fx[..., None], fy[..., None], fxx[..., None], fyy[..., None], fxy[..., None]), axis=2)

    # Cross-Terms for all adjacent channels
    for adj_ch in range(len(db_chs)):
        # Early exit
        if np.sum(lambda_cross_ch) <= np.finfo(float).eps:
            break

        # Continue for current channel
        if adj_ch == ch or db_chs[adj_ch]['K'] is None:
            continue
        adjChImg = db_chs[adj_ch]['Image']  # Curr cross channel

        # Compute cross terms
        dxf = np.array([[1, 0, -1]])
        dyf = np.array([[1], [0], [-1]])

        diag_term = convolve2d(adjChImg, np.fliplr(np.flipud(dxf)), mode='full')
        diag_term = diag_term[:, 1:] * f
        conv_term = convolve2d(f, np.fliplr(np.flipud(dxf)), mode='full')
        Sxf = (lambda_cross_ch[adj_ch] * 0.5) * (adjChImg * conv_term[:, 1:] - diag_term)

        diag_term = convolve2d(adjChImg, np.fliplr(np.flipud(dyf)), mode='full')
        diag_term = diag_term[1:, :] * f
        conv_term = convolve2d(f, np.fliplr(np.flipud(dyf)), mode='full')
        Syf = (lambda_cross_ch[adj_ch] * 0.5) * (adjChImg * conv_term[1:, :] - diag_term)

        # Gather
        Kmultf = np.concatenate((Kmultf, np.stack((Sxf, Syf), axis=2)), axis=2)

        return Kmultf
    




def solve_fft(Nomin1, Denom1, tau, lambda_, f):
    """
    Solves Ax = b with
    A = (tau*lambda* K'* K + eye ) and b = tau * lambda * K' * B + f 
    
    Parameters:
    Nomin1 (ndarray): F(K)'*F(y)
    Denom1 (ndarray): |F(K)|.^2
    tau (float): scaling parameter
    lambda_ (float): regularization parameter
    f (ndarray): input image
    
    Returns:
    x (ndarray): output image after solving Ax = b
    """
    # Compute denominator and numerator terms
    denom = tau * 2 * lambda_ * Denom1 + 1
    numer = tau * 2 * lambda_ * Nomin1 + fft2(f)
    
    # Solve Ax = b using FFT
    x = ifft2(numer / denom).real
    return x

def pd_channel_deconv(channels, ch, x_0, db_chs, lambda_residual, lambda_cross_ch, lambda_tv,
                      lambda_black, max_it, tol, tol_offset, verbose, iterate_fig):
    # Shortcut for the TV norm.
    Amplitude = lambda u: np.sqrt(u ** 2)
    F = lambda u: np.sum(np.sum(Amplitude(u)))

    # Prepare
    sizey = channels[ch].Image.shape
    otfk = fftshift(np.real(ifft2(channels[ch].K, sizey)))
    Nomin1 = np.conj(otfk) * fft2(channels[ch].Image)
    Denom1 = np.abs(otfk) ** 2

    # Prox operator
    # L1 norm
    ProxFS = lambda u, l: u / np.maximum(1, Amplitude(u))

    # Huber norm
    # alpha = 0.01
    # ProxFS = lambda u,sigma: (u/(1 + alpha*sigma)) ./ max(1, Amplitude((u/(1 + alpha*sigma))));

    # Fast data solve with fft
    ProxG = lambda f, tau, l: solve_fft(Nomin1, Denom1, tau, l, f)

    # Minimization weights
    L = compute_operator_norm(lambda x: Kmult(x, ch, db_chs, lambda_cross_ch, lambda_tv),
                              lambda x: KSmult(x, ch, db_chs, lambda_cross_ch, lambda_tv),
                              sizey)
    sigma = 1.0
    tau = 0.7 / (sigma * L ** 2)
    theta = 1.0

    # Set initial iterate
    if x_0 is None:
        x_0 = channels[ch].Image
    f = x_0
    g = Kmult(f, ch, db_chs, lambda_cross_ch, lambda_tv)
    f1 = f

    # Display it.
    if verbose == 'all':
        fig = plt.figure(iterate_fig)
        plt.clf()
        plt.imshow(f1, cmap='gray'), plt.axis('image')
        plt.title(f'Local PD iterate {0}')
        plt.pause(0.1)

    for i in range(max_it):

        fold = f
        g = ProxFS(g + sigma * Kmult(f1, ch, db_chs, lambda_cross_ch, lambda_tv), sigma)
        f = ProxG(f - tau * KSmult(g, ch, db_chs, lambda_cross_ch, lambda_tv), tau, lambda_residual)
        f1 = f + theta * (f-fold)

        # Display iteration
        if verbose == 'all' and i % 20 == 0:
            plt.figure(iterate_fig)
            plt.clf()
            plt.imshow(f1, cmap='gray')
            plt.axis('image')
            plt.title('Local PD iterate {}'.format(i))
            plt.pause(0.1)

        diff = (f + tol_offset) - (fold + tol_offset)
        f_comp = (f + tol_offset)
        if verbose == 'brief' or verbose == 'all':
            print('Ch: {}, iter {}, diff {:5.5g}'.format(ch, i, np.linalg.norm(diff.ravel(),2) / np.linalg.norm(f_comp.ravel(),2)))
        if np.linalg.norm(diff.ravel(),2) / np.linalg.norm(f_comp.ravel(),2) < tol:
            break

    res = f1
    return res


def residual_pd_deconv(channels, db_chs, ch, w_res_curr, w_tv_curr, w_black_curr, w_cross_curr, res_iter, tol, max_it, verbose, iterate_fig, local_iterate_fig):
    detail_tol = tol
    
    for d in range(res_iter + 1):
        if d == 0:
            channels_res_blur = channels
            x_0 = db_chs[ch].Image
            tol_offset = np.zeros_like(db_chs[ch].Image)
        else:
            channels_res_blur = channels
            channels_res_blur[ch].Image = channels[ch].Image - convolve2d(db_chs[ch].Image, db_chs[ch].K, mode='same')
            channels_res_blur[ch].Image = 1 + channels_res_blur[ch].Image
            x_0 = channels_res_blur[ch].Image
            w_res_curr = w_res_curr * 3.0
            tol_offset = db_chs[ch].Image - 1
        
        verbose_local = 'brief'
        x = pd_channel_deconv(channels_res_blur, ch, x_0, db_chs, w_res_curr, w_cross_curr, w_tv_curr, w_black_curr, max_it, detail_tol, tol_offset, verbose_local, local_iterate_fig)
        x[x < 1] = 1
        
        if d == 0:
            db_chs[ch].Image = x
        else:
            db_chs[ch].Image = db_chs[ch].Image + (x - 1)
            db_chs[ch].Image[db_chs[ch].Image < 0] = 0
        
        if verbose == 'all':
            plt.figure(iterate_fig[ch])
            if d > 0:
                plt.subplot(1, res_iter + 1, 1)
            plt.imshow(db_chs[ch].Image - 1)
            plt.title(f'Startup iterate in Ch {ch}')
            if d > 0:
                plt.subplot(1, res_iter + 1, 1 + (d - 1))
                plt.imshow(x, cmap='gray', aspect='equal')
                plt.axis('off')
                plt.title(f'Detail layer {d-1} in Ch {ch} with lambda_res {w_res_curr:.5g}')
            plt.pause(0.1)
    
    res = db_chs[ch].Image
    return res



def pd_joint_deconv(channels, channels_0, w_base, max_it, tol, verbose):

    # Check for channel sanity
    if len(channels) < 1:
        raise ValueError('No valid channels found for deconvolution.')
    if channels_0 is not None and len(channels_0) != len(channels):
        raise ValueError('Initial channels do not match channels.')
        
    # Initialize all channels
    if channels_0 is None:
        channels_0 = channels
    db_chs = channels.copy()
    iterate_fig = []  # Debug figures for all channels
    
    for ch in range(len(channels)):
        # Set to the initial iterate if necessary
        if db_chs[ch]['K'].size != 0:
            db_chs[ch]['Image'] = channels_0[ch]['Image']
        elif db_chs[ch]['K'].shape[0] == db_chs[ch]['K'].shape[1] and \
                db_chs[ch]['K'].shape[0] % 2 == 1:
            raise ValueError('Blur size is not square or odd.')  # Blur size check
        
        # Result figure
        if verbose == 'all':
            iterate_fig.append(plt.figure())
            
    
    # Do startup minimization
    for s in range(w_base.shape[0]):
        if verbose == 'brief' or verbose == 'all':
            print(f'\n### Startup iteration {s+1} ###\n')
        
        # Get current channel and weights
        ch_opt = w_base[s, 0].astype(int)
        w_res_curr = w_base[s, 1]
        w_tv_curr = w_base[s, 2]
        w_black_curr = w_base[s, 3]
        w_cross_curr = w_base[s, 4:-1]
        res_iter = w_base[s, -1]
        
        # Check for residual deconvolution sanity
        if res_iter > 0 and not np.array_equal(w_cross_curr, np.zeros(w_cross_curr.shape)):
            raise ValueError('Residual iteration with cross channel terms is not supported')
        
        # edgetaper to better handle circular boundary conditions
        ks = channels[ch_opt]['K'].shape[0]
        for ch in range(len(channels)):
            channels[ch]['Image'] = np.pad(channels[ch]['Image'], ((ks, ks), (ks, ks)), 'edge')
            db_chs[ch]['Image'] = np.pad(db_chs[ch]['Image'], ((ks, ks), (ks, ks)), 'edge')
            for _ in range(4):
                channels[ch]['Image'] = edgetaper(channels[ch]['Image'], channels[ch_opt]['K'])
                db_chs[ch]['Image'] = edgetaper(db_chs[ch]['Image'], channels[ch_opt]['K'])
        
            channels[ch]['Image'] += 1.0
            db_chs[ch]['Image'] += 1.0
        
        # Do residual pd deconvolution
        db_chs[ch_opt]['Image'] = residual_pd_deconv(channels, db_chs, ch_opt, w_res_curr, w_tv_curr, w_black_curr, w_cross_curr, res_iter, tol, max_it, verbose, iterate_fig)  
    
        for ch in range(len(channels)):
            # remove padding
            channels[ch]['Image'] = channels[ch]['Image'][ks:-ks, ks:-ks] - 1.0
            db_chs[ch]['Image'] = db_chs[ch]['Image'][ks:-ks, ks:-ks] - 1.0
        
    return db_chs
