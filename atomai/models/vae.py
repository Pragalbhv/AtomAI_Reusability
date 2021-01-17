"""
vae.py
=======

Module for analysis of system "building blocks"" with variational autoencoders

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""

import os
from copy import deepcopy as dc
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torchvision.utils import make_grid

from ..losses_metrics import (joint_rvae_loss, joint_vae_loss, rvae_loss,
                              vae_loss)
from ..nets import init_VAE_nets
from ..trainers import viBaseTrainer
from ..utils import (crop_borders, extract_subimages, get_coord_grid,
                     imcoordgrid, set_train_rng, subimg_trajectories,
                     transform_coordinates)


class BaseVAE(viBaseTrainer):
    """
    General class for encoder-decoder type of deep latent variable models

    Args:
        in_dim:
            (height, width) or (height, width, channel) of input images
        latent_dim:
            Number of latent dimensions
        nb_classes:
            Number of classes (for class-conditional VAEs)
        seed:
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **conv_decoder (bool):
            use convolutional layers in decoder (doesn't apply to  rVAE)
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units OR conv filters in decoder (Default: 128)
    """
    def __init__(self,
                 in_dim: Tuple[int],
                 latent_dim: int,
                 nb_classes: int = 0,
                 coord: int = 0,
                 discrete_dim: Optional[List] = None,
                 seed: int = 0,
                 **kwargs: Union[int, bool]) -> None:
        super(BaseVAE, self).__init__()
        """
        Initializes encoder-decoder object
        """

        in_dim_error_msg = (
            "You must specify the input dimensions and pass them as a tuple. "
            "For images, specify (height, width) or (height, width, channels)" +
            " if multiple channels. For spectra, specify (length,)")

        if in_dim is None or not isinstance(in_dim, (tuple, list)):
            raise AssertionError(in_dim_error_msg)
        if isinstance(in_dim, tuple) and not isinstance(in_dim[0], int):
            raise AssertionError(in_dim_error_msg)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        set_train_rng(seed)
        np.random.seed(seed)

        self.in_dim = in_dim
        self.z_dim = latent_dim
        if isinstance(discrete_dim, list):
            self.z_dim = self.z_dim + sum(discrete_dim)
        self.discrete_dim = discrete_dim

        if coord:
            if len(in_dim) not in (2, 3):
                raise NotImplementedError(
                    "VAE with rotation and translational invariance are " +
                    "available only for 2D image data")
            self.z_dim = latent_dim + coord
        self.nb_classes = nb_classes

        (encoder_net, decoder_net,
         self.metadict) = init_VAE_nets(
            in_dim, latent_dim, coord, discrete_dim, nb_classes, **kwargs)
        self.set_model(encoder_net, decoder_net)

        self.coord = coord

    def encode_(self,
                x_new: Union[np.ndarray, torch.Tensor],
                **kwargs: int) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder

        Args:
            x_test:
                image array to encode
            **num_batches:
                number of batches (Default: 10)

        Returns:
            Concatenated array of encoded vectors
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                encoded = self.encoder_net(x_i)
            encoded = torch.cat(encoded, -1).cpu().numpy()
            return encoded

        if isinstance(x_new, np.ndarray):
            x_new = torch.from_numpy(x_new).float()
        if (x_new.ndim == len(self.in_dim) == 2 or
           x_new.ndim == len(self.in_dim) == 3):
            x_new = x_new.unsqueeze(0)
        x_new = x_new.to(self.device)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(x_new) // num_batches
        z_encoded = np.zeros((x_new.shape[0], self.z_dim))
        for i in range(num_batches):
            x_i = x_new[i*batch_size:(i+1)*batch_size]
            z_encoded_i = inference()
            z_encoded[i*batch_size:(i+1)*batch_size] = z_encoded_i
        x_i = x_new[(i+1)*batch_size:]
        if len(x_i) > 0:
            z_encoded_i = inference()
            z_encoded[(i+1)*batch_size:] = z_encoded
        return z_encoded

    def encode(self,
               x_new: Union[np.ndarray, torch.Tensor],
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder

        Args:
            x_test:
                image array to encode
            **num_batches:
                number of batches (Default: 10)

        Returns:
            Mean and SD of the encoded distribution(s)
        """
        z = self.encode_(x_new, **kwargs)
        z_mean = z[:self.z_dim]
        z_logsd = z[self.z_dim:]
        return z_mean, z_logsd

    def decode(self, z_sample: Union[np.ndarray, torch.Tensor],
               y: Optional[Union[int, np.ndarray, torch.Tensor]] = None
               ) -> np.ndarray:
        """
        Takes a point in latent space and maps it to data space
        via the learned generative model

        Args:
            z_sample: point(s) in latent space
            y: label

        Returns:
            Generated ("decoded") image(s)
        """

        if isinstance(z_sample, np.ndarray):
            z_sample = torch.from_numpy(z_sample).float()
        if len(z_sample.size()) == 1:
            z_sample = z_sample[None, ...]
        if self.coord:
            x_coord = self.x_coord.expand(z_sample.size(0), *self.x_coord.size())
        z_sample = z_sample.cuda() if torch.cuda.is_available() else z_sample
        if y is not None:
            if isinstance(y, int):
                y = torch.tensor(y)
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            y = y.cuda() if torch.cuda.is_available() else y
            targets = to_onehot(y, self.nb_classes)
            z_sample = torch.cat((z_sample, targets), dim=-1)
        if torch.cuda.is_available():
            self.decoder_net.cuda()
        self.decoder_net.eval()
        with torch.no_grad():
            if self.coord:
                x_decoded = self.decoder_net(x_coord, z_sample)
            else:
                x_decoded = self.decoder_net(z_sample)
        imdec = x_decoded.cpu().numpy()
        return imdec

    def forward_(self,
                 x_new: Union[np.ndarray, torch.Tensor],
                 **kwargs: int) -> np.ndarray:
        """
        Forward prediction with uncertainty quantification by sampling from
        the encoded mean and std. Works only for regular VAE (and not for rVAE)

        Args:
            x_new:
                image array to encode
            **num_samples:
                number of samples to generate from normal distribution

        Returns:
            Ensemble of "decoded" images
        """
        num_samples = kwargs.get("num_samples", 32)
        if isinstance(x_new, np.ndarray):
            x_new = torch.from_numpy(x_new).float()
        if torch.cuda.is_available():
            x_new = x_new.cuda()
            self.encoder_net.cuda()
        with torch.no_grad():
            z_mean, z_logsd = self.encoder_net(x_new)
        z_sd = torch.exp(z_logsd)
        ndist = torch.distributions.Normal(z_mean, z_sd)
        decoded_all = []
        for i in range(num_samples):
            z_sample = ndist.rsample()
            z_sample = z_sample.view(1, -1)
            decoded_all.append(self.decode_(z_sample))
        decoded_all = np.concatenate(decoded_all, axis=0)
        return decoded_all

    def encode_images(self,
                      imgdata: np.ndarray,
                      **kwargs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes every pixel of every image in image stack

        Args:
            imgdata:
                3D numpy array. Can also be a single 2D image
            **num_batches (int):
                number of batches for for encoding pixels of a single image

        Returns:
            Cropped original image stack and encoded array (cropping is due to finite window size)
        """

        if (imgdata.ndim == len(self.in_dim) == 2 or
           imgdata.ndim == len(self.in_dim) == 3):
            imgdata = np.expand_dims(imgdata, axis=0)
        imgdata_encoded, imgdata_ = [], []
        for i, img in enumerate(imgdata):
            print("\rImage {}/{}".format(i+1, imgdata.shape[0]), end="")
            img_, img_encoded = self.encode_image_(img, **kwargs)
            imgdata_encoded.append(img_encoded)
            imgdata_.append(img_)
        return np.array(imgdata_), np.array(imgdata_encoded)

    def encode_image_(self,
                      img: np.ndarray,
                      **kwargs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops and encodes a subimage around each pixel in the input image.
        The size of subimage is determined by size of images in VAE training data.

        Args:
            img:
                2D numpy array
            **num_batches (int):
                number of batches for encoding subimages

        Returns:
            Cropped original image and encoded array (cropping is due to finite window size)
        """

        num_batches = kwargs.get("num_batches", 10)
        inf = np.int(1e5)
        img_to_encode = img.copy()
        coordinates = get_coord_grid(img_to_encode, 1, return_dict=False)
        batch_size = coordinates.shape[0] // num_batches
        encoded_img = -inf * np.ones((*img_to_encode.shape, self.z_dim))
        for i in range(num_batches):
            coord_i = coordinates[i*batch_size:(i+1)*batch_size]
            subimgs_i, com_i, _ = extract_subimages(
                                    img_to_encode, coord_i, self.in_dim[0])
            if len(subimgs_i) > 0:
                z_mean, _ = self.encode(subimgs_i, num_batches=10)
                for k, (l, m) in enumerate(com_i):
                    encoded_img[int(l), int(m)] = z_mean[k]
        coord_i = coordinates[(i+1)*batch_size:]
        if len(coord_i) > 0:
            subimgs_i, com_i, _ = extract_subimages(
                                    img_to_encode, coord_i, self.in_dim[0])
            if len(subimgs_i) > 0:
                z_mean, _ = self.encode(subimgs_i, num_batches=10)
                for k, (l, m) in enumerate(com_i):
                    encoded_img[int(l), int(m)] = z_mean[k]

        img_to_encode[encoded_img[..., 0] == -inf] = 0
        img_to_encode = crop_borders(img_to_encode[..., None], 0)
        encoded_img = crop_borders(encoded_img, -inf)

        return img_to_encode[..., 0], encoded_img

    def encode_trajectories(self,
                            imgdata: np.ndarray,
                            coord_class_dict: Dict[int, np.ndarray],
                            window_size: int,
                            min_length: int,
                            rmax: int,
                            **kwargs: int
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculates trajectories and latent variable value
        for each point in a trajectory.

        Args:
            imgdata:
                NN output (preferable) or raw data
            coord_class_dict:
                atomic/defect/particle coordinates
            window_size:
                size of subimages to crop
            min_length:
                minimum length of trajectory to be included
            rmax:
                maximum allowed distance (projected on xy plane) between defect
                in one frame and the position of its nearest neigbor in the next one
            **num_batches (int):
                number of batches for self.encode (Default: 10)

        Returns:
            List of encoded trajectories and corresponding movie frame numbers
        """
        t = subimg_trajectories(
                imgdata, coord_class_dict, window_size, min_length, rmax)
        trajectories, frames, subimgs_all = t.get_all_trajectories()
        trajectories_enc_all = []
        for traj, subimgs in zip(trajectories, subimgs_all):
            z_mean, _ = self.encode(
                subimgs, num_batches=kwargs.get("num_batches", 10))
            traj_enc = np.concatenate((traj[:, :2], z_mean), axis=-1)
            trajectories_enc_all.append(traj_enc)
        return trajectories_enc_all, frames, subimgs_all

    def manifold2d(self, **kwargs: Union[int, List, str, bool]) -> None:  # use torchvision's grid here
        """
        Performs mapping from latent space to data space allowing the learned
        manifold to be visualized. This works only for 2d latent variable
        (not counting angle & translation dimensions)

        Args:
            **d (int): grid size
            **l1 (list): range of 1st latent variable
            **l2 (list): range of 2nd latent variable
            **label(int): label in class-conditioned (r)VAE
            **cmap (str): color map (Default: gnuplot)
            **draw_grid (bool): plot semi-transparent grid
            **origin (str): plot origin (e.g. 'lower')
        """
        y = kwargs.get("label")
        if y is None and self.nb_classes != 0:
            y = 0
        elif y and self.nb_classes == 0:
            y = None
        l1, l2 = kwargs.get("l1"), kwargs.get("l2")
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        if len(self.in_dim) == 2:
            figure = np.zeros((self.in_dim[0] * d, self.in_dim[1] * d))
        elif len(self.in_dim) == 3:
            figure = np.zeros((self.in_dim[0] * d, self.in_dim[1] * d, self.in_dim[-1]))
        if l1 and l2:
            grid_x = np.linspace(l1[0], l1[1], d)
            grid_y = np.linspace(l2[0], l2[1], d)
        else:
            grid_x = norm.ppf(np.linspace(0.05, 0.95, d))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                if y is not None:
                    imdec = self.decode(z_sample, y)
                else:
                    imdec = self.decode(z_sample)
                figure[i * self.in_dim[0]: (i + 1) * self.in_dim[0],
                       j * self.in_dim[1]: (j + 1) * self.in_dim[1]] = imdec
        if figure.min() < 0:
            figure = (figure - figure.min()) / figure.ptp()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(figure, cmap=cmap, origin=kwargs.get("origin", "lower"))
        draw_grid = kwargs.get("draw_grid")
        if draw_grid:
            major_ticks_x = np.arange(0, d * self.in_dim[0], self.in_dim[0])
            major_ticks_y = np.arange(0, d * self.in_dim[1], self.in_dim[1])
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
            ax.grid(which='major', alpha=0.6)
        if not kwargs.get("savefig"):
            plt.show()
        else:
            savedir = kwargs.get("savedir", './vae_learning/')
            fname = kwargs.get("filename", "manifold_2d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            fig.savefig(os.path.join(savedir, '{}.png'.format(fname)))
            plt.close(fig)

    def manifold_traversal(self, cont_idx: int,
                           d: int = 10,
                           cont_idx_fixed: int = 0,
                           plot: bool = True,
                           **kwargs: Union[str, float]
                           ) -> np.ndarray:
        """
        Latent space traversals for joint continuous and discrete
        latent representations
        """
        if self.discrete_dim is None:
            raise TypeError(
                "Traversal of latent space is implemented only for joint",
                " continuous and discrete latent distributions")
        num_samples = d**2
        cont_dim = self.z_dim - self.discrete_dim
        # Get continuous latent coordinates
        samples_cont = np.zeros(shape=(num_samples, cont_dim)) + cont_idx_fixed
        cdf_traversal = np.linspace(0.05, 0.95, d)
        cont_traversal = norm.ppf(cdf_traversal)
        for i in range(d):
            for j in range(d):
                samples_cont[i * d + j, cont_idx] = cont_traversal[j]
        # Get discrete latent coordinates
        disc_dim = self.discrete_dim[0]
        samples_disc = []
        for i in range(d):
            samples_disc_i = np.zeros((d, disc_dim))
            samples_disc_i[:, i] = 1
            samples_disc.append(samples_disc_i)
        samples_disc = np.concatenate(samples_disc)
        # Put them together and pass through a decoder
        samples = np.concatenate((samples_cont, samples_disc), -1)
        decoded = self.decode(samples)
        # Use a built-in torchvision utility to construct a nice grid
        grid = make_grid(torch.from_numpy(decoded)[:, None],
                         nrow=d, padding=kwargs.get("pad", 2)).numpy()
        if len(self.in_dim) == 2:  # for grayscale images take a single channel
            grid = grid[0]
        if plot:
            plt.figure(figsize=(12, 12))
            plt.imshow(grid, cmap='gnuplot',
                       origin=kwargs.get("origin", "lower"))
        return grid

    @classmethod
    def visualize_manifold_learning(cls,
                                    frames_dir: str,
                                    **kwargs: Union[str, int]) -> None:
        """
        Creates and stores a video showing evolution of
        learned 2D manifold during rVAE's training

        Args:
            frames_dir:
                directory with snapshots of manifold as .png files
                (the files should be named as "1.png", "2.png", etc.)
            **moviename (str): name of the movie
            **frame_duration (int): duration of each movie frame
        """
        from atomai.utils import animation_from_png
        movie_name = kwargs.get("moviename", "manifold_learning")
        duration = kwargs.get("frame_duration", 1)
        animation_from_png(frames_dir, movie_name, duration, remove_dir=False)

    def elbo_fn(self, x: torch.Tensor, x_reconstr: torch.Tensor,
                *args: torch.Tensor) -> torch.Tensor:
        """
        Calculates ELBO
        """
        return vae_loss(self.loss, self.in_dim, x, x_reconstr, *args)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        VAE's forward pass with training/test loss computation
        """
        x = x.to(self.device)
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)

        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd)

    def _check_inputs(self,
                      X_train: np.ndarray,
                      y_train: Optional[np.ndarray] = None,
                      X_test: Optional[np.ndarray] = None,
                      y_test: Optional[np.ndarray] = None
                      ) -> None:
        """
        Asserts that dimensionality and number classes contained in
        training and test data matches those specified at initialization
        """
        if self.in_dim != X_train.shape[1:]:
            raise RuntimeError(
                "The values of input dimensions you specified do not match " +
                "the training data dimensions. " +
                "Expected {} but got {}".format(self.in_dim, X_train.shape[1:]))
        if X_test is not None and self.in_dim != X_test.shape[1:]:
            raise RuntimeError(
                "The values of input dimensions you specified do not match " +
                "the test data dimensions. " +
                "Expected {} but got {}".format(self.in_dim, X_test.shape[1:]))
        if y_train is not None and self.nb_classes == 0:
            raise RuntimeError(
                "You must have forgotten to specify number of classes " +
                "during the initialization. Example of correct usage: " +
                "vae = VAE(in_dim=(28, 28), nb_classes=10)); " +
                "vae.fit(train_data, train_labels).")
        lbl_match = True
        if y_train is not None and y_test is None:
            lbl_match = self.nb_classes == len(np.unique(y_train))
        elif y_train is not None and y_test is not None:
            lbl_match = (self.nb_classes == len(np.unique(y_train))
                         == len(np.unique(y_test)))
        if not lbl_match:
            raise RuntimeError(
                "The number of classes specified at initialization must be " +
                "equal the the number of classes in train and test labels")

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "mse",
            **kwargs) -> None:
        """
        Trains VAE model

        Args:
            X_train:
                For images, 3D or 4D stack of training images with dimensions
                (n_images, height, width) for grayscale data or
                or (n_images, height, width, channels) for multi-channel data.
                For spectra, 2D stack of spectra with dimensions (length,)
            X_test:
                3D or 4D stack of test images or 2D stack of spectra with
                the same dimensions as for the X_train (Default: None)
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of training images/spectra
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of test images/spectra
            loss:
                reconstruction loss function, "ce" or "mse" (Default: "mse")
            **filename (str):
                file path for saving model aftereach training cycle ("epoch")
        """
        self._check_inputs(X_train, y_train, X_test, y_test)
        self.compile_trainer(
            (X_train, y_train), (X_test, y_test), **kwargs)
        self.loss = loss  # this part needs to be handled better

        for e in range(self.training_cycles):
            self.current_epoch = e
            elbo_epoch = self.train_epoch()
            self.loss_history["train_loss"].append(elbo_epoch)
            if self.test_iterator is not None:
                elbo_epoch_test = self.evaluate_model()
                self.loss_history["test_loss"].append(elbo_epoch_test)
            self.print_statistics(e)
            self.save_model(self.filename)
        return

    def _2torch(self,
                X: Union[np.ndarray, torch.Tensor],
                y: Union[np.ndarray, torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Rules for conversion of numpy arrays to torch tensors
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        return X, y

    def print_statistics(self, e):
        """
        Prints training and (optionally) test loss after each training cycle
        """
        if self.test_iterator is not None:
            template = 'Epoch: {}/{}, Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -self.loss_history["train_loss"][-1],
                  -self.loss_history["test_loss"][-1]))
        else:
            template = 'Epoch: {}/{}, Training loss: {:.4f}'
            print(template.format(e+1, self.training_cycles,
                  -self.loss_history["train_loss"][-1]))


class VAE(BaseVAE):
    """
    Implements a standard Variational Autoencoder (VAE)

    Args:
        in_dim:
            Input dimensions for image data passed as (heigth, width)
            for grayscale data or (height, width, channels)
            for multichannel data
        latent_dim:
            Number of VAE latent dimensions
        nb_classes:
            Number of classes for class-conditional rVAE
        seed:
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units in decoder (Default: 128)

    Example:

    >>> input_dim = (28, 28) # Input data dimensions (without n_samples)
    >>> # Intitialize model
    >>> vae = aoi.models.VAE(input_dim)
    >>> # Train
    >>> vae.fit(imstack_train, training_cycles=100, batch_size=100)
    >>> # Visualize learned manifold (for 2 latent dimesnions)
    >>> vae.manifold2d(origin="upper", cmap="gnuplot2)

    One can also pass labels to train a class-conditioned VAE

    >>> # Intitialize model
    >>> vae = aoi.models.VAE(input_dim, nb_classes=10)
    >>> # Train
    >>> vae.fit(imstack_train, labels_train, training_cycles=100, batch_size=100)
    >>> # Visualize learned manifold for class 1
    >>> vae.manifold2d(label=1, origin="upper", cmap="gnuplot2")
    """
    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        super(VAE, self).__init__(in_dim, latent_dim, nb_classes, 0, **kwargs)
        set_train_rng(seed)


class rVAE(BaseVAE):
    """
    Implements rotationally and translationally invariant
    Variational Autoencoder (VAE) based on the idea of "spatial decoder"
    by Bepler et al. in arXiv:1909.11663. In addition, this class allows
    implementating the class-conditioned VAE and skip-VAE (arXiv:1807.04863)
    with rotational and translational variance.

    Args:
        in_dim:
            Input dimensions for image data passed as (heigth, width)
            for grayscale data or (height, width, channels)
            for multichannel data
        latent_dim:
            Number of VAE latent dimensions associated with image content
        nb_classes:
            Number of classes for class-conditional rVAE
        translation:
            account for xy shifts of image content (Default: True)
        seed:
            seed for torch and numpy (pseudo-)random numbers generators
        **conv_encoder (bool):
            use convolutional layers in encoder
        **numlayers_encoder (int):
            number of layers in encoder (Default: 2)
        **numlayers_decoder (int):
            number of layers in decoder (Default: 2)
        **numhidden_encoder (int):
            number of hidden units OR conv filters in encoder (Default: 128)
        **numhidden_decoder (int):
            number of hidden units in decoder (Default: 128)
        **skip (bool):
            uses generative skip model with residual paths between
            latents and decoder layers (Default: False)

    Example:

    >>> input_dim = (28, 28)  # intput dimensions
    >>> # Intitialize model
    >>> rvae = aoi.models.rVAE(input_dim)
    >>> # Train
    >>> rvae.fit(imstack_train, training_cycles=100,
                 batch_size=100, rotation_prior=np.pi/2)
    >>> rvae.manifold2d(origin="upper", cmap="gnuplot2")

    One can also pass labels to train a class-conditioned rVAE

    >>> # Intitialize model
    >>> rvae = aoi.models.rVAE(input_dim, nb_classes=10)
    >>> # Train
    >>> rvae.fit(imstack_train, labels_train, training_cycles=100,
    >>>            batch_size=100, rotation_prior=np.pi/2)
    >>> # Visualize learned manifold for class 1
    >>> rvae.manifold2d(label=1, origin="upper", cmap="gnuplot2")
    """

    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 translation: bool = True,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initializes rVAE model
        """
        coord = 3 if translation else 1  # xy translations and/or rotation
        args = (in_dim, latent_dim, nb_classes, coord)
        super(rVAE, self).__init__(*args, **kwargs)
        set_train_rng(seed)
        self.x_coord = None
        self.translation = translation
        self.dx_prior = None
        self.phi_prior = None
        self.anneal_dict = None

    def elbo_fn(self,
                x: torch.Tensor,
                x_reconstr: torch.Tensor,
                *args: torch.Tensor,
                **kwargs: float
                ) -> torch.Tensor:
        """
        Computes ELBO
        """
        return rvae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        rVAE's forward pass with training/test loss computation
        """
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        if mode == "eval":
            with torch.no_grad():
                z_mean, z_logsd = self.encoder_net(x)
        else:
            z_mean, z_logsd = self.encoder_net(x)
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)
        phi = z[:, 0]  # angle
        if self.translation:
            dx = z[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z = z[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z = z[:, 1:]  # image content

        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)

        x_coord_ = transform_coordinates(x_coord_, phi, dx)
        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)
        # KL annealing terms
        b1 = b2 = 1
        if isinstance(self.anneal_dict, dict):
            e_ = self.current_epoch
            b1 = self.anneal_dict["kl_im"]
            b2 = self.anneal_dict["kl_rot"]
            b1 = b1[-1] if len(b1) < e_ + 1 else b1[e_]
            b2 = b2[-1] if len(b2) < e_ + 1 else b2[e_]
        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd,
                            phi_prior=self.phi_prior, b1=b1, b2=b2)

    def fit(self,
            X_train: Optional[Union[np.ndarray, torch.Tensor]],
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "mse",
            **kwargs) -> None:
        """
        Trains rVAE model

        Args:
            X_train:
                3D or 4D stack of training images with dimensions
                (n_images, height, width) for grayscale data or
                or (n_images, height, width, channels) for multi-channel data
            X_test:
                3D or 4D stack of test images with the same dimensions
                as for the X_train (Default: None)
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of training images
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of test images
            loss:
                reconstruction loss function, "ce" or "mse" (Default: "mse")
            **translation_prior (float):
                translation prior
            **rotation_prior (float):
                rotational prior
            **filename (str):
                file path for saving model aftereach training cycle ("epoch")
            **recording (bool):
                saves a learned 2d manifold at each training step
        """
        self._check_inputs(X_train, y_train, X_test, y_test)
        self.x_coord = imcoordgrid(X_train.shape[1:]).to(self.device)
        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.phi_prior = kwargs.get("rotation_prior", 0.1)
        self.anneal_dict = kwargs.get("anneal_dict")
        self.compile_trainer(
            (X_train, y_train), (X_test, y_test), **kwargs)
        self.loss = loss  # this part needs to be handled better
        self.recording = kwargs.get("recording", False)

        for e in range(self.training_cycles):
            self.current_epoch = e
            elbo_epoch = self.train_epoch()
            self.loss_history["train_loss"].append(elbo_epoch)
            if self.test_iterator is not None:
                elbo_epoch_test = self.evaluate_model()
                self.loss_history["test_loss"].append(elbo_epoch_test)
            self.print_statistics(e)
            if self.recording and self.z_dim in [3, 5]:
                self.manifold2d(savefig=True, filename=str(e))
            self.save_model(self.filename)
        if self.recording and self.z_dim in [3, 5]:
            self.visualize_manifold_learning("./vae_learning")


class jVAE(BaseVAE):

    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 discrete_dim: List[int] = [1],
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initializes rVAE model
        """
        args = (in_dim, latent_dim, 0, 0, discrete_dim)
        super(jVAE, self).__init__(*args, **kwargs)
        set_train_rng(seed)
        self.kdict_ = dc(kwargs)
        self.kdict_["num_iter"] = 0

    def elbo_fn(self,
                x: torch.Tensor,
                x_reconstr: torch.Tensor,
                *args: torch.Tensor,
                **kwargs: Union[List, int]
                ) -> torch.Tensor:
        """
        Computes ELBO
        """
        return joint_vae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        JointVAE's forward pass with training/test loss computation
        """
        tau = self.kdict_.get("temperature", .67)
        x = x.to(self.device)
        if mode == "eval":
            with torch.no_grad():
                latent_ = self.encoder_net(x)
        else:
            latent_ = self.encoder_net(x)
            self.kdict_["num_iter"] += 1
        z_mean, z_logsd = latent_[:2]
        z_sd = torch.exp(z_logsd)
        z_cont = self.reparameterize(z_mean, z_sd)

        alphas = latent_[2:]
        z_disc = [self.reparameterize_discrete(a, tau) for a in alphas]
        z_disc = torch.cat(z_disc, 1)

        z = torch.cat((z_cont, z_disc), dim=1)

        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)

        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(z)
        else:
            x_reconstr = self.decoder_net(z)

        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd, alphas, **self.kdict_)

    def encode(self,
               x_new: Union[np.ndarray, torch.Tensor],
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder

        Args:
            x_new:
                image array to encode
            **num_batches:
                number of batches (Default: 10)

        Returns:
            Mean and SD of the encoded continuous distribution,
            and alphas ("class probabilities") for the encoded discrete distribution(s)
        """
        z_encoded = self.encode_(x_new, **kwargs)
        cont_dim = self.z_dim - self.discrete_dim
        z_mean = z_encoded[:cont_dim]
        z_sd = z_encoded[cont_dim:cont_dim+cont_dim]
        alphas = z_encoded[cont_dim+cont_dim:]
        return z_mean, z_sd, alphas


class jrVAE(BaseVAE):
    """
    Rotationally-invariant VAE for joint continuous and
    discrete latent variables.
    """
    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 translation: bool = True,
                 discrete_dim: List[int] = [1],
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initializes joint rVAE model (jrVAE)
        """
        coord = 3 if translation else 1  # xy translations and/or rotation
        args = (in_dim, latent_dim, nb_classes, coord, discrete_dim)
        super(jrVAE, self).__init__(*args, **kwargs)
        set_train_rng(seed)
        self.x_coord = None
        self.translation = translation
        self.dx_prior = None
        self.phi_prior = None
        self.anneal_dict = None
        self.kdict_ = dc(kwargs)
        self.kdict_["num_iter"] = 0

    def elbo_fn(self,
                x: torch.Tensor,
                x_reconstr: torch.Tensor,
                *args: torch.Tensor,
                **kwargs: Union[List, int]
                ) -> torch.Tensor:
        """
        Computes ELBO
        """
        return joint_rvae_loss(self.loss, self.in_dim, x, x_reconstr, *args, **kwargs)

    def forward_compute_elbo(self,
                             x: torch.Tensor,
                             y: Optional[torch.Tensor] = None,
                             mode: str = "train"
                             ) -> torch.Tensor:
        """
        Joint rVAE's forward pass with training/test loss computation
        """
        tau = self.kdict_.get("temperature", .67)
        x_coord_ = self.x_coord.expand(x.size(0), *self.x_coord.size())
        x = x.to(self.device)
        if mode == "eval":
            with torch.no_grad():
                latent_ = self.encoder_net(x)
        else:
            latent_ = self.encoder_net(x)
            self.kdict_["num_iter"] += 1

        z_mean, z_logsd = latent_[:2]
        z_sd = torch.exp(z_logsd)
        z_cont = self.reparameterize(z_mean, z_sd)
        phi = z_cont[:, 0]  # angle
        if self.translation:
            dx = z_cont[:, 1:3]  # translation
            dx = (dx * self.dx_prior).unsqueeze(1)
            z_cont = z_cont[:, 3:]  # image content
        else:
            dx = 0  # no translation
            z_cont = z_cont[:, 1:]  # image content
        x_coord_ = transform_coordinates(x_coord_, phi, dx)

        alphas = latent_[2:]
        z_disc = [self.reparameterize_discrete(a, tau) for a in alphas]
        z_disc = torch.cat(z_disc, 1)

        z = torch.cat((z_cont, z_disc), dim=1)

        if y is not None:
            targets = to_onehot(y, self.nb_classes)
            z = torch.cat((z, targets), -1)

        if mode == "eval":
            with torch.no_grad():
                x_reconstr = self.decoder_net(x_coord_, z)
        else:
            x_reconstr = self.decoder_net(x_coord_, z)

        return self.elbo_fn(x, x_reconstr, z_mean, z_logsd, alphas, **self.kdict_)

    def fit(self,
            X_train: Optional[Union[np.ndarray, torch.Tensor]],
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "mse",
            **kwargs) -> None:
        """
        Trains rVAE model

        Args:
            X_train:
                3D or 4D stack of training images with dimensions
                (n_images, height, width) for grayscale data or
                or (n_images, height, width, channels) for multi-channel data
            X_test:
                3D or 4D stack of test images with the same dimensions
                as for the X_train (Default: None)
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of training images
            y_train:
                Vector with labels of dimension (n_images,), where n_images
                is a number of test images
            loss:
                reconstruction loss function, "ce" or "mse" (Default: "mse")
            **translation_prior (float):
                translation prior
            **rotation_prior (float):
                rotational prior
            **filename (str):
                file path for saving model aftereach training cycle ("epoch")
            **recording (bool):
                saves a learned 2d manifold at each training step
        """
        self._check_inputs(X_train, y_train, X_test, y_test)
        self.x_coord = imcoordgrid(X_train.shape[1:]).to(self.device)
        self.dx_prior = kwargs.get("translation_prior", 0.1)
        self.phi_prior = kwargs.get("rotation_prior", 0.1)
        self.anneal_dict = kwargs.get("anneal_dict")
        self.compile_trainer(
            (X_train, y_train), (X_test, y_test), **kwargs)
        self.loss = loss  # this part needs to be handled better
        self.recording = kwargs.get("recording", False)

        for e in range(self.training_cycles):
            self.current_epoch = e
            elbo_epoch = self.train_epoch()
            self.loss_history["train_loss"].append(elbo_epoch)
            if self.test_iterator is not None:
                elbo_epoch_test = self.evaluate_model()
                self.loss_history["test_loss"].append(elbo_epoch_test)
            self.print_statistics(e)
            self.save_model(self.filename)

    def encode(self,
               x_new: Union[np.ndarray, torch.Tensor],
               **kwargs: int) -> Tuple[np.ndarray]:
        """
        Encodes input image data using a trained VAE's encoder

        Args:
            x_new:
                image array to encode
            **num_batches:
                number of batches (Default: 10)

        Returns:
            Mean and SD of the encoded continuous distribution,
            and alphas ("class probabilities") for the encoded discrete distribution(s)
        """
        z_encoded = self.encode_(x_new, **kwargs)
        cont_dim = self.z_dim - self.discrete_dim
        z_mean = z_encoded[:cont_dim]
        z_sd = z_encoded[cont_dim:cont_dim+cont_dim]
        alphas = z_encoded[cont_dim+cont_dim:]
        return z_mean, z_sd, alphas


def to_onehot(idx: torch.Tensor, n: int) -> torch.Tensor: # move to utils!
    """
    One-hot encoding of label
    """
    if torch.max(idx).item() >= n:
        raise AssertionError(
            "Labelling must start from 0 and "
            "maximum label value must be less than total number of classes")
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    onehot = torch.zeros(idx.size(0), n, device=device_)
    onehot.scatter_(1, idx, 1)
    return onehot
