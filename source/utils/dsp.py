from scipy.signal import resample

def resample_audio(audio, orig_sampling_rate, target_sampling_rate):
    if orig_sampling_rate == target_sampling_rate:
        return audio
    duration = len(audio) / orig_sampling_rate
    num_samples = int(duration * target_sampling_rate)
    return resample(audio, num_samples).astype('float32')