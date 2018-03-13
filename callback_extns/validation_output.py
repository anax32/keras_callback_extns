from keras.callbacks import Callback
from keras.layers import Conv2D, Dense

import numpy as np
from scipy.misc import imsave as save_image

from os.path import join
from os import makedirs

class ValidationOutput (Callback):
  """Obtain some validation data from the validation generator and
  run the model prediction over this data.
  Write the output to a given directory, under epoch number.
  Write the validation scores with data filename to a csv in the dir"""
  def __init__ (self, out_dir, validation_generator, validation_steps):
    # Callback.params contains training parameters
    # Callback.model contains curent model
    self.dir = out_dir
    self.data = validation_generator
    self.count = int (validation_steps)

  def on_epoch_end (self, epoch, logs=None):
    """Run prediction over n items,
    store inputs and outputs in <dir>/<epoch_number>,
    store scores in <dir>/<epoch_number>/scores.csv"""
    epoch_path = join (self.dir, str (epoch).zfill (4))

    try:
      makedirs (epoch_path)
    except IOError:
      pass

    for i in range (0, self.count):
      X, _ = next (self.data)

      for j in range (0, X.shape[0]):
        data_path = join (epoch_path, "%04i_%04i_in.png" % (i, j))
        save_image (data_path, X[j,:,:,0])

      Y = self.model.predict (X, batch_size = X.shape[0])

      # for multiple outputs, assume the first output is an image...
      if type (Y) is type ([]):
        for idx, y in enumerate (Y):
          for j in range (0, y.shape[0]):
            pred_path = join (epoch_path, "%04i_%04i_pred_%02i.png" % (i, j, idx))
            save_image (pred_path, y[j,:,:,0])
      else:
        for j in range (0, Y.shape[0]):
          pred_path = join (epoch_path, "%04i_%04i_pred.png" % (i, j))
          save_image (pred_path, Y[j,:,:,0])