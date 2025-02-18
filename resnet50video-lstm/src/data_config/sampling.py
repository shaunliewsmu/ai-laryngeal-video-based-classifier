import numpy as np
from src.config.config import SEED

class VideoSampling:
    @staticmethod
    def uniform_sampling(total_frames, num_samples):
        """Uniformly sample frames across the video"""
        indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
        return indices
    
    @staticmethod
    def random_sampling(total_frames, num_samples):
        """Randomly sample frames from the video with reproducible results"""
        rng = np.random.RandomState(SEED)
        indices = sorted(rng.choice(total_frames, num_samples, replace=False))
        return indices
    
    @staticmethod
    def sliding_window_sampling(total_frames, num_samples, window_size=None):
        """Sample frames using sliding window approach"""
        if window_size is None:
            window_size = total_frames // num_samples
        stride = (total_frames - window_size) // (num_samples - 1) if num_samples > 1 else 1
        indices = [min(i * stride, total_frames - 1) for i in range(num_samples)]
        return indices

    @staticmethod
    def get_sampler(sampling_method):
        sampling_methods = {
            'uniform': VideoSampling.uniform_sampling,
            'random': VideoSampling.random_sampling,
            'sliding': VideoSampling.sliding_window_sampling
        }
        return sampling_methods.get(sampling_method, VideoSampling.uniform_sampling)
