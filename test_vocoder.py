from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile
import torch

from ddsp.vocoder import load_model, Audio2Mel
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from preprocessing.process_pipeline import get_pitch_parselmouth, get_pitch_crepe
from utils.hparams import set_hparams, hparams


def vocode(audio_path, crepe=False):
    wav, mel = NsfHifiGAN.wav2spec(audio_path)
    if crepe:
        f0, pitch_coarse = get_pitch_crepe(wav, mel, hparams)
    else:
        f0, pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
    return mel, f0


if __name__ == "__main__":
    config_path = "./configs/config_nsf.yaml"
    hparams = set_hparams(config=config_path, exp_name='', infer=True, reset=True, hparams_str='', print_hparams=False)
    vocoder = NsfHifiGAN()
    audio_path = "./raw/2211202558_23.wav"
    nsf_audio_path = f"./nsf_{Path(audio_path).name}"
    mel, f0 = vocode(audio_path)
    nsf_audio = vocoder.spec2wav(mel, f0=f0)
    soundfile.write(nsf_audio_path, nsf_audio, 44100, format="wav")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    model, args = load_model("./model_6000.pt", device=device)

    sampling_rate = 44100
    hop_length = 512
    win_length = 2048
    n_mel_channels = 160
    mel_fmin = 0
    mel_fmax = 24000

    # load input
    x, _ = librosa.load(audio_path, sr=sampling_rate)
    x_t = torch.from_numpy(x).float().to(device)
    x_t = x_t.unsqueeze(0).unsqueeze(0)  # (T,) --> (1, 1, T)

    # mel analysis
    mel_extractor = Audio2Mel(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        n_mel_channels=n_mel_channels,
        win_length=win_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax).to(device)

    mel = mel_extractor(x_t)

    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=hop_length / sampling_rate, voicing_threshold=0.6,
        pitch_floor=65, pitch_ceiling=800).selected_array['frequency']
    pad_size = (int(len(x) // hop_length) - len(f0) + 1) // 2
    f0 = np.pad(f0, [[pad_size, mel.size(1) - len(f0) - pad_size]], mode='constant')

    #
    uv = f0 == 0
    f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

    # key change
    f0 = f0 * 2 ** (float(0) / 12)

    # forward
    with torch.no_grad():
        signal, _, (s_h, s_n) = model(mel, f0, max_upsample_dim=32)
        signal = signal.squeeze().cpu().numpy()
        soundfile.write(f"./ddsp_{Path(audio_path).name}", signal, sampling_rate)
