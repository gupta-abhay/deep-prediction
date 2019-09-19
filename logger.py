import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TensorLogger(object):
    # creating file in given logdir ... defaults to ./runs/
    def __init__(self, _logdir='./runs/'):
        if not os.path.exists(_logdir):
            os.makedirs(_logdir)
        
        self.writer = SummaryWriter(log_dir=_logdir)

    # adding scalar value to tb file
    def scalar_summary(self, _tag, _value, _step):
        self.writer.add_scalar(_tag, _value, _step)

    # adding image value to tb file
    def image_summary(self, _tag, _image, _step, _format='CHW'):
        """
            default dataformat for image tensor is (3, H, W)
            can be changed to
                : (1, H, W) - dataformat = CHW
                : (H, W, 3) - dataformat HWC
                : (H, W) - datformat HW
        """
        # 
        self.writer.add_image(_tag, _image, _step, dataformat=_format)

    # adding matplotlib figure to tb file
    def figure_summary(self, _tag, _figure, _step):
        self.writer.add_figure(_tag, _figure, _step)

    # adding video to tb file
    def video_summary(self, _tag, _video, _step, _fps=4):
        """
            default torch fps is 4, can be changed
            also, video tensor should be of format (N, T, C, H, W)
            values should be between [0,255] for unit8 and [0,1] for float32
        """
        # default value of video fps is 4 - can be changed
        self.writer.add_video(_tag, _video, _step, _fps)

    # adding audio to tb file
    def audio_summary(self, _tag, _sound, _step, _sampleRate = 44100):
        """
            default torch sample rate is 44100, can be changed
            also, sound tensor should be of format (1, L)
            values should lie between [-1,1]
        """
        self.writer.add_audio(_tag, _sound, _step, sample_rate = _sampleRate)

    # adding text to tb file
    def text_summary(self, _tag, _textString, _step):
        self.writer.add_text(_tag, _textString, _step)

    # adding histograms to tb file
    def histogram_summary(self, _tag, _histogram, _step, _bins='tensorflow'):
        self.writer.add_histrogram(_tag, _histogram, _step, bins=_bins)
