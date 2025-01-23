# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Ensure the image is a numpy array
                if isinstance(img, np.ndarray):
                    # Add channel dimension if necessary
                    if img.ndim == 2:  # Shape (height, width)
                        img = np.expand_dims(img, axis=-1)  # Shape (height, width, 1)
                    
                    # Add batch dimension
                    img = np.expand_dims(img, axis=0)  # Shape (1, height, width, channels)

                # Write the image to TensorBoard
                tf.summary.image(f"{tag}/{i}", img, step=step)

            self.writer.flush()
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        # Directly log the histogram using TensorFlow's summary API
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step)
            self.writer.flush()