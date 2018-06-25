import tensorflow as tf
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO
import scipy.misc

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self,tag, batch, step):
        """ Log a list of images"""
        img_summaries=[]

        for i, img in enumerate(batch):
            try:
                s=StringIO()
            except:
                s=BytesIO()
            scipy.misc.toimage(img).save(s,format="png")

            img_sum=tf.Summary.Image(encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1])

            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
        summary=tf.Summary(value=img_summaries)
        self.writer.add_summary(summary,step)
