from keras.callbacks import Callback
from keras.layers import Conv2D, Dense

import numpy as np
from scipy.misc import imsave as save_image

from os.path import join

class WeightWriter (Callback):
  """convert weight matrices to images and write them to a folder.
  Currently writes weights for Conv2D and Dense layers.
  Padding is added around Conv2D kernels to get a grid shape.
  Alignment of the images is such that the x-axis represents inputs
  from the previous layer, and the y-axis represents parameters of the layer.
  
  i.e., for a Conv2D weights image:
    ---> inputs --->
   |  x x x x x x x
  \/  x x x x x x x
   f  x x x x x x x
   i  x x x x x x x
   t  x x x x x x x
   e  x x x x x x x
   r  x x x x x x x
   |  x x x x x x x
   \/ x x x x x x x
   
   Similarly for a dense layer, inputs along the top, outputs along the side."""   
  def __init__ (self, out_dir):
    # Callback.params contains training parameters
    # Callback.model contains curent model
    self.dir = out_dir

  @staticmethod    
  def Conv2D_weights_to_image (layer):
    """convert the kernels of a conv2d layer to an image"""
    W, b = layer.get_weights ()

    # remap the weights to 0..1
    # FIXME: sometimes we have constraints on a layer,
    # should we use those values here if they exist?
    W = (W - np.min (W)) / (np.max (W)-np.min(W))
    # add a border
    # TODO: add border formatting parameters to the object?
    W_ = np.pad (W, [(1,0), (1,0), (0,0), (0,0)], "constant", constant_values=(1,1))
    # flatten to 2d tiles
    return W_.swapaxes (0, 3).swapaxes (1, 3).reshape ((W_.shape[0]*W_.shape[3], W_.shape[1]*W_.shape[2]))
  
  @staticmethod  
  def Dense_weights_to_image (layer):
    """convert the weight matrix of a dense layer to an image"""
    W, b = layer.get_weights ()
    W = W.T # transpose to get inputs at top, params on side
    return (W - np.min (W))/(np.max (W) - np.min (W))

  @staticmethod
  def model_weights_to_images (model):
    """convert all the layer-weights of a model to a list of images.
    NB some layers do not have weights and are not imaged.
    TODO: Conv3D
    returns a list of pairs (name, image)"""
    handlers = {
      Conv2D: WeightWriter.Conv2D_weights_to_image,
      Dense: WeightWriter.Dense_weights_to_image
    }

    return [(l.name, handlers[type (l)] (l)) for l in model.layers if type (l) in handlers]

  def on_epoch_end (self, epoch, logs=None):
    """Calls model_weights_to_images and writes the images to
    self.dir\<layer_name>_<epoch number>.png"""
    images = WeightWriter.model_weights_to_images (self.model)
    
    # write the images
    for name, image in images:
      save_image (join (self.dir, ".".join ([name,  str (epoch), "png"])), image)