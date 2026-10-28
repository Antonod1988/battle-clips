"""Announcer lines: offline Windows speech (System.Speech, SAPI voices) shaped with ffmpeg into a big-hall announcer.

`line(text, style)` returns a stereo array at dsp.SR; results are cached as WAV files next to the module
(clips/audio/.cache, not committed). A style is a voice name, SSML prosody and an ffmpeg filter chain; add your own
to STYLES for another clip.
"""
import hashlib
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

import dsp

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import FFMPEG  # noqa: E402  (the one place the ffmpeg path is set)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

STYLES = {
    # the arena's announcer: a man's voice slowed and pitched down, warm, squeezed, in a big stone hall
    'announcer': {
        'voice': 'Microsoft David Desktop', 'rate': '-22%', 'pitch': '-12%', 'hall': (2.8, 0.9, 0.42),
        'chain': 'asetrate=48000*0.80,aresample=48000,atempo=1.12,highpass=f=70,'
                 'equalizer=f=120:t=q:w=1:g=5,equalizer=f=2800:t=q:w=1.2:g=3,equalizer=f=6500:t=q:w=1:g=-4,'
                 'acompressor=threshold=-22dB:ratio=6:attack=4:release=160:makeup=7,'
                 'asoftclip=type=tanh:param=1.2'},
    # Scorpion: a woman's voice dropped far down and doubled an octave lower, rough and close
    'scorpion': {
        'voice': 'Microsoft Zira Desktop', 'rate': '-5%', 'pitch': '+0%', 'hall': (0.9, 0.25, 0.22),
        'chain': 'asplit=2[a][b];[a]asetrate=48000*0.62,aresample=48000,atempo=1.28[a1];'
                 '[b]asetrate=48000*0.50,aresample=48000,atempo=1.59[b1];[a1][b1]amix=inputs=2:weights=1 0.65,'
                 'highpass=f=80,lowpass=f=5200,volume=3,asoftclip=type=tanh:param=1.8,'
                 'acompressor=threshold=-20dB:ratio=5:attack=3:release=120:makeup=5'},
}


def _speak(text, style, wav):
    st = STYLES[style]
    ssml = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<prosody rate="{st["rate"]}" pitch="{st["pitch"]}" volume="100">{text}</prosody></speak>')
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{st['voice']}'); "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{wav}', $f); $s.SpeakSsml(@'\n{ssml}\n'@); $s.SetOutputToNull()")
    subprocess.check_call(['powershell', '-NoProfile', '-Command', ps])


def _hall(dur, decay, wet, seed):
    imp = dsp.ir(dur, decay, damp=5000, rng=np.random.default_rng(seed), pre=0.025)
    return imp, wet


def line(text, style='announcer'):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f'{style}|{text}|{STYLES[style]}'.encode()).hexdigest()[:16]
    out = os.path.join(CACHE, f'{style}-{key}.wav')
    if not os.path.exists(out):
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, 'raw.wav')
            _speak(text, style, raw)
            dry = os.path.join(td, 'dry.wav')
            chain = STYLES[style]['chain']
            # trim the engine's silence, then the style's chain (pitch, EQ, compression, grit)
            pre = 'silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=1:stop_threshold=-45dB:stop_duration=0.15'
            if '[a]' in chain:
                fc = f'[0:a]{pre},{chain}[o]'
            else:
                fc = f'[0:a]{pre},{chain}[o]'
            subprocess.check_call([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', raw, '-filter_complex', fc,
                                   '-map', '[o]', '-ar', '48000', '-ac', '1', '-c:a', 'pcm_f32le', dry])
            # the hall: the dry voice convolved with a long stone-hall response (ffmpeg's afir)
            hall_dur, hall_decay, wet = STYLES[style]['hall']
            imp = dsp.ir(hall_dur, hall_decay, damp=5200, rng=np.random.default_rng(len(text)), pre=0.03)
            irf = os.path.join(td, 'ir.wav')
            wavfile.write(irf, dsp.SR, (imp / np.max(np.abs(imp)) * 0.5).astype(np.float32))
            pad = hall_dur + 0.3
            subprocess.check_call([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', dry, '-i', irf,
                                   '-filter_complex',
                                   f'[0:a]apad=pad_dur={pad},pan=stereo|c0=c0|c1=c0[v];'
                                   f'[v][1:a]afir=dry={10 * (1 - wet):.2f}:wet={10 * wet:.2f}:irnorm=1,'
                                   'alimiter=limit=0.89[o]',
                                   '-map', '[o]', '-ar', '48000', '-c:a', 'pcm_f32le', out])
    sr, y = wavfile.read(out)
    y = y.astype(np.float64)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.9
