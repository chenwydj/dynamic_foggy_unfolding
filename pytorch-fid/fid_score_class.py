#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=4,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
'''
    64: 0,   # First max pooling features
    192: 1,  # Second max pooling featurs
    768: 2,  # Pre-aux classifier features
    2048: 3  # Final average pooling features
'''
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--classN', default=19, type=int,
                    help='number of class to use')

def one_hot(index, classes, cuda):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)#[:, 1:, :, :]


def get_activations(files_img, files_label, model, classN, batch_size=50, dims=2048, cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files_img) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files_img):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files_img)

    n_batches = len(files_img) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((classN, n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32) for f in files_img[start:end]])
        labels = np.array([imread(str(f)).astype(np.uint8) for f in files_label[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch_label = torch.from_numpy(labels).type(torch.LongTensor)
        if cuda:
            batch = batch.cuda()
            batch_label = batch_label.cuda()
        
        if dims == 2048:
            '''last fully connected feature vector'''
            if (batch_label == 255).nonzero().size(0) > 0:
                batch_label[batch_label == 255] = -1; batch_label += 1
                batch_label = one_hot(batch_label, classN+1, cuda)[:, 1:, :, :]
            else:
                batch_label = one_hot(batch_label, classN, cuda)

            for c in range(classN):
                pred = model(batch * batch_label[:, c:c+1, :, :])[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred_arr[c, start:end, :] = pred.cpu().data.numpy().reshape(batch_size, -1)
        else:
            pred = model(batch)[0]
            _, _, h, w = pred.size()
            batch_label = F.upsample(batch_label.type(torch.cuda.FloatTensor).unsqueeze(1), size=(h, w), mode='nearest')[:, 0, :, :].type(torch.cuda.LongTensor)
            if (batch_label == 255).nonzero().size(0) > 0:
                batch_label[batch_label == 255] = -1; batch_label += 1
                batch_label = one_hot(batch_label, classN+1, cuda)[:, 1:, :, :]
            else:
                batch_label = one_hot(batch_label, classN, cuda)

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            for c in range(classN):
                pred_c = adaptive_avg_pool2d(pred * batch_label[:, c:c+1, :, :], output_size=(1, 1)) * h * w / (batch_label[:, c:c+1, :, :].view(batch_size, -1).sum(1) + 0.01)
                pred_arr[c, start:end, :] = pred_c.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(files_img, files_label, model, classN, batch_size=50, dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    activations = get_activations(files_img, files_label, model, classN, batch_size, dims, cuda, verbose)
    mus = [np.mean(activations[c], axis=0) for c in range(classN)]
    sigmas = [np.cov(activations[c], rowvar=False) for c in range(classN)]
    return mus, sigmas


def _compute_statistics_of_path(path_img, path_label, model, batch_size, dims, cuda, classN):
    if path_img.endswith('.npz'):
        f = np.load(path_img)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        files_img = list(pathlib.Path(path_img).glob('*.jpg'))
        files_label = list(pathlib.Path(path_label).glob('*.png'))
        m, s = calculate_activation_statistics(files_img, files_label, model, classN, batch_size, dims, cuda)

    return m, s


def calculate_frechet_distance(mu1s, sigma1s, mu2s, sigma2s, classN, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    
    fid_values = []
    for c in tqdm(range(classN)):
        mu1 = np.atleast_1d(mu1s[c])
        mu2 = np.atleast_1d(mu2s[c])

        sigma1 = np.atleast_2d(sigma1s[c])
        sigma2 = np.atleast_2d(sigma2s[c])

        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid_values.append(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid_values


def calculate_fid_given_paths(paths, batch_size, cuda, dims, classN):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1s, s1s = _compute_statistics_of_path(paths[0], paths[1], model, batch_size, dims, cuda, classN)
    m2s, s2s = _compute_statistics_of_path(paths[2], paths[3], model, batch_size, dims, cuda, classN)
    fid_values = calculate_frechet_distance(m1s, s1s, m2s, s2s, classN)

    return fid_values


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_values = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims,
                                          args.classN)
    print('FID: ', fid_values)
