from keras.callbacks import Callback
from keras.layers import Conv2D, Dense

import tensorflow as tf

class TFRunMetaData (Callback):
  """example usage:
    from keras.callbacks import TensorBoard as TensorBoardCallback
    
    # create the tensorboard callback
    tb = TensorBoardCallback (logdir="logs")
    
    # create the run metadatacallback with a reference to
    # the tensorboard callback
    rmdc = TFRunMetaData (tb)

    # compile the model and associate the run_metadata
    # parameter with the run_metadata member variable
    model.compile (
      optimizer = "adam",
      loss = "rms",
      metrics = ["accuracy"]
      options = rmdc.run_options,
      run_metadata = rmdc.run_metadata
      )
    
    # fit the model with the callbacks
    model.fit (images, masks,
               epochs=64,
               validation_split=0.3,
               callbacks = [tb, rmdc])
               
    Note on usage in TensorBoard:
    You will need libcupti to be in the <CUDA_PATH>\bin folder, usually this is
    found in the <CUDA_PATH>\extras folder.
    If you recieve an error about libcupti methods not being loaded, it is probably
    due to this.
    
    Sometimes, Tensorboard refuses to show the Compute Time or Memory stats for a
    graph. First, ensure you have selected an item in the "Session Runs" dropdown.
    Second, ensure your device is selected in the devices checkbox list (headed
    "Devices included in stats:"), if in doubt, just tick everything!
    Third, refresh the view and try those options in a different order, it seems
    a little fussy.
  """
  def __init__ (self, tensorboard_callback, trace_level=tf.RunOptions.FULL_TRACE):
    """initialise the baseclass,
    keep a reference to the tensorboard callback (we access the tensorboard writer)
    create run_options and run_metadata objects to record data."""
    Callback.__init__ (self)
    self.tensorboard_callback = tensorboard_callback
    self.run_options = tf.RunOptions (trace_level=trace_level)
    self.run_metadata = tf.RunMetadata ()

  def on_epoch_begin (self, epoch, logs):
    """clears the run_metadata object.
    We clear it here, rather than on_epoch_end so it initialises correctly"""
    #print ("called RUNMETADATACALLBACK begin epoch " + str (epoch))
    self.run_metadata.Clear ()

  def on_epoch_end (self, epoch, logs):
    """writes the current run_metadata object to the tensorboard writer.
    Flushes the tensorboard writer."""
    #print ("called RUNMETADATACALLBACK on epoch " + str (epoch))
    #print ("  run_metadata.size = " + str (self.run_metadata.ByteSize ()))

    # create a new rundata for the next epoch
    self.tensorboard_callback.writer.add_run_metadata (
      self.run_metadata,
      "epoch_%d" % (epoch))
    self.tensorboard_callback.writer.flush ()