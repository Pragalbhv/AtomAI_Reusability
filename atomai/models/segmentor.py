from typing import Type, Union, Tuple, Optional, Dict
import torch
import numpy as np
from ..trainers import SegTrainer
from ..predictors import SegPredictor
from ..transforms import seg_augmentor
from ..utils import get_downsample_factor

#additional depedency:- sklearn possibly implement our own f1?
from sklearn.metrics import f1_score


class Segmentor(SegTrainer):
    """
    Model for semantic segmentation-based analysis of images

    Args:
        model:
            Type of model to train: 'Unet', 'ResHedNet' or 'dilnet' (Default: 'Unet').
            See atomai.nets for more details. One can also pass here a custom
            fully convolutional neural network model.
        nb_classes:
            Number of classes in classification scheme (last NN layer)
        **batch_norm (bool):
            Apply batch normalization after each convolutional layer
            (Default: True)
        **dropout (bool):
            Apply dropouts to the three inner blocks in the middle of a network
            (Default: False)
        **upsampling_mode (str):
            "bilinear" or "nearest" upsampling method (Default: "bilinear")
        **nb_filters (int):
            Number of convolutional filters (aka "kernels") in the first
            convolutional block (this number doubles in the consequtive block(s),
            see definition of *Unet* and *dilnet* models for details)
        **with_dilation (bool):
            Use dilated convolutions in the bottleneck of *Unet*
            (Default: False)
        **layers (list):
            List with a number of layers in each block.
            For *UNet* the first 4 elements in the list
            are used to determine the number of layers
            in each block of the encoder (including bottleneck layer),
            and the number of layers in the decoder  is chosen accordingly
            (to maintain symmetry between encoder and decoder)

    Example:

    >>> # Initialize model
    >>> model = aoi.models.Segmentor(nb_classes=3)
    >>> # Train
    >>> model.fit(images, masks, images_test, masks_test,
    >>>        training_cycles=300, compute_accuracy=True, swa=True)
    >>> # Predict with trained model
    >>> nn_output, coordinates = model.predict(expdata)
    """
    def __init__(self,
                 model: Type[Union[str, torch.nn.Module]] = "Unet",
                 nb_classes: int = 1,
                 **kwargs) -> None:
        super(Segmentor, self).__init__(model, nb_classes, **kwargs)
        self.downsample_factor = None
        #self.binary_thresh= None shifted to trainer.py for saving

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = 'ce',
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            training_cycles: int = 1000,
            batch_size: int = 32,
            compute_accuracy: bool = False,
            full_epoch: bool = False,
            swa: bool = False,
            perturb_weights: bool = False,
            **kwargs):
        """
        Compiles a trainer and performs model training

        Args:
            X_train:
                4D numpy array or pytorch tensor of training images
                (n_samples, 1, height, width). One can also pass a regular
                3D image stack without a channel dimension of 1 which will
                be added automatically
            y_train:
                4D (binary) / 3D (multiclass) numpy array or pytorch tensor
                of training masks (aka ground truth) stacked along
                the first dimension. The reason why in the multiclass case
                the X_train is 4-dimensional and the y_train is 3-dimensional
                is because of how the cross-entropy loss is calculated in PyTorch
                (see https://pytorch.org/docs/stable/nn.html#nllloss).
            X_test:
                4D numpy array or pytorch tensor of test images
                (stacked along the first dimension)
            y_test:
                4D (binary) / 3D (multiclass) numpy array or pytorch tensor
                of test masks (aka ground truth) stacked along
                the first dimension.
            loss:
                loss function. Available loss functions are: 'mse' (MSE),
                'ce' (cross-entropy), 'focal' (focal loss; single class only),
                and 'dice' (dice loss)
            optimizer:
                weights optimizer (defaults to Adam optimizer with lr=1e-3)
            training_cycles: Number of training 'epochs'.
                If full_epoch argument is set to False, 1 epoch == 1 mini-batch.
                Otherwise, each cycle corresponds to all mini-batches of data
                passing through a NN.
            batch_size: Size of training and test mini-batches
            compute_accuracy:
                Computes accuracy function at each training cycle
            full_epoch:
                If True, passes all mini-batches of training/test data
                at each training cycle and computes the average loss. If False,
                passes a single (randomly chosen) mini-batch at each cycle.
            swa:
                Saves the recent stochastic weights and averages
                them at the end of training
            perturb_weights:
                Time-dependent weight perturbation, :math:`w\\leftarrow w + a / (1 + e)^\\gamma`,
                where parameters *a* and *gamma* can be passed as a dictionary,
                together with parameter *e_p* determining every *n*-th epoch at
                which a perturbation is applied
            **lr_scheduler (list of floats):
                List with learning rates for each training iteration/epoch.
                If the length of list is smaller than the number of training iterations,
                the last values in the list is used for the remaining iterations.
            **print_loss (int):
                Prints loss every *n*-th epoch
            **accuracy_metrics (str):
                Accuracy metrics (used only for printing training statistics)
            **filename (str):
                Filename for model weights
                (appended with "_test_weights_best.pt" and "_weights_final.pt")
            **plot_training_history (bool):
                Plots training and test curves vs. training cycles
                at the end of training
            **auto_thresh (bool):
                Performs automatic binary threshold selection for optimal f1-score
            **ES (bool):
                Early stopping mode on/off
            **patience (int):
                 patience for early stopping
            **tolerance (float):
                 tolerance for early stopping
            **weight_decay (float):
                  weight decay for model 
            **kwargs:
                One can also pass kwargs for utils.datatransform class
                to perform the augmentation "on-the-fly" (e.g. rotation=True,
                gauss_noise=[20, 60], etc.)
        """
        
        do_auto_thresh=kwargs.get("auto_thresh",False)
        self.compile_trainer(
            (X_train, y_train, X_test, y_test),
            loss, optimizer, training_cycles, batch_size,
            compute_accuracy, full_epoch, swa, perturb_weights,
            **kwargs)
        

        self.augment_fn = seg_augmentor(self.nb_classes, **kwargs)
        _ = self.run()
        
        if do_auto_thresh and X_test is not None and y_test is not None:
            self.auto_thresh_predict(X_test, y_test)
            self.save_model(self.filename + "_metadict_final")
            
            
        
        

    def predict(self,
                imgdata: Union[np.ndarray, torch.Tensor],
                refine: bool = False,
                logits: bool = True,
                resize: Tuple[int, int] = None,
                compute_coords: bool = True,
                **kwargs) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Apply (trained) model to new data

        Args:
            image_data:
                3D image stack or a single 2D image (all greyscale)
            refine:
                Atomic positions refinement with 2d Gaussian peak fitting
                (may take some time)
            logits:
                Indicates whether the features are passed through
                a softmax/sigmoid layer at the end of a neural network
                (logits=True for AtomAI models)
            resize:
                Resizes input data to (new_height, new_width) before passing
                to a neural network
            compute_coords (bool):
                Computes centers of the mass of individual blobs
                in the segmented images (Default: True)
            **thresh (float):
                Value between 0 and 1 for thresholding the NN output
                (Default: 0.5)
            **d (int):
                half-side of a square around each atomic position used
                for refinement with 2d Gaussian peak fitting. Defaults to 1/4
                of average nearest neighbor atomic distance
            **num_batches (int): number of batches (Default: 10)
            **norm (bool): Normalize data to (0, 1) during pre-processing
            **verbose (bool): verbosity

        Returns:
            Semantically segmented image and coordinates of (atomic) objects

        """
        if self.downsample_factor is None:
            self.downsample_factor = get_downsample_factor(self.net)
        use_gpu = self.device == 'cuda'
        ##################################start of edit#######################################################
        print(not kwargs.get('thresh', False))
        if self.binary_thresh and not kwargs.get('thresh', False) :
            print('Performing auto-thresh prediction')
            prediction = SegPredictor(
                self.net, refine, resize, use_gpu, logits,
                nb_classes=self.nb_classes, downsampling=self.downsample_factor,thresh=self.binary_thresh,
                **kwargs).run(imgdata, compute_coords, **kwargs)
            
            class_pred = np.zeros(prediction[0].shape[:3])
            class_pred[prediction[0][:,:,:,0] > self.binary_thresh] = 1
            prediction=prediction + (class_pred,)
            
        else:
            prediction = SegPredictor(
                self.net, refine, resize, use_gpu, logits,
                nb_classes=self.nb_classes, downsampling=self.downsample_factor,
                **kwargs).run(imgdata, compute_coords, **kwargs)
       ##################################end of edit#######################################################
       
        

        return prediction
    
    ##################################start of edit#######################################################
    
    def auto_thresh_predict(self, images_val,labels_val):
        '''
        Performs automatic binary thresholding on validation set -> stores best threshold to model
        
        Args:
            images_val:
                4D numpy array or pytorch tensor of validation images
                (stacked along the first dimension)
            labels_val:
                4D (binary) / 3D (multiclass) numpy array or pytorch tensor
                of validation masks (aka ground truth) stacked along
                the first dimension.
        
        '''
        print('Calculating Automatic Threshold')
        
        pred_val = self.predict(images_val)
        best_thresold = 0.5
        max_f1=-1
        thresh=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        for idx in range(len(thresh)):
            print('Checking Threshold item#: ',idx+1,'/',len(thresh), end="\r")
            
            threshold = thresh[idx]
            class_pred = np.zeros(pred_val[0].shape[:3])

            class_pred[pred_val[0][:,:,:,0] > threshold] = 1

            f_this=f1_score(labels_val.ravel(), class_pred.ravel())

            if f_this>max_f1:
                best_threso=threshold
                max_f1=f_this

        self.binary_thresh=best_threso
   ##################################end of edit#######################################################
        



    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights dictionary
        """
        weight_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(weight_dict)
