# Author(s): Dr. Patrick Lemoine

import numpy as np
from scipy.io.wavfile import write

def generate_smooth_trajectory(start_theta, start_phi, end_theta, end_phi, samples):
    """
    Generates smoothed theta and phi values using a cosine interpolation
    to mimic a gentle arc (no zigzag).
    """
    t = np.linspace(0, 1, samples)
    theta_traj = start_theta + (end_theta - start_theta) * (1 - np.cos(t * np.pi)) / 2
    phi_traj   = start_phi + (end_phi - start_phi) * (1 - np.cos(t * np.pi)) / 2
    return theta_traj, phi_traj

def generate_flying_drone_frame_wav(filename="drone_audio_signal_frame.wav", 
                              n_mics=16, fs=16000, duration_s=2.0, 
                              f_drone=1000, noise_level=0.02,
                              theta_path=(20,160), phi_path=(20,340)):
    samples = int(fs * duration_s)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    audio = np.zeros((samples, n_mics))
    # Generate smooth trajectory
    theta_traj, phi_traj = generate_smooth_trajectory(theta_path[0], phi_path[0], theta_path[1], phi_path[1], samples)
    mic_angles = []
    for i in range(n_mics):
        th = 180 * (i + 0.5) / n_mics      # microphone inclination
        ph = 360 * (i + 0.5) / n_mics      # microphone azimuth
        mic_angles.append((th, ph))
    for i in range(n_mics):
        mic_th, mic_ph = mic_angles[i]
        phase_delay = np.radians(theta_traj - mic_th) + np.radians(phi_traj - mic_ph)
        base_signal = np.sin(2 * np.pi * f_drone * t + phase_delay)
        base_signal *= np.hanning(samples)
        noise = noise_level * np.random.randn(samples)
        audio[:, i] = base_signal + noise
    # Normalize and convert for WAV
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(filename, fs, audio_int16)
    print(f"Realistic drone flight .wav generated: {filename} ({duration_s:.1f}s, {n_mics} mics, {f_drone}Hz)")


def generate_flying_drone_wav(file_path, n_mics=16, fs=16000, duration_s=0.1, freq=1000, noise_level=0.02, phase_step=0.1):
    """
    Generates a multi-track sinusoidal signal (motor/drone or radio/pilot) with noise.
    file_path: WAV file name
    freq: frequency in Hz (1000 for drone, 2450 for radio pilot)
    phase_step: phase variation between microphones to simulate spatial origin
    """
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    signals = []
    for i in range(n_mics):
        phase_shift = i * phase_step
        base_signal = np.sin(2 * np.pi * freq * t + phase_shift)
        base_signal *= np.hanning(len(base_signal))
        noise = noise_level * np.random.randn(len(t))
        signals.append(base_signal + noise)
    audio = np.array(signals).T  # (samples, n_mics)
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(file_path, fs, audio_int16)
    print(f"Fake signal saved to {file_path} (freq={freq}Hz, {fs}Hz, {n_mics} mics, {duration_s*1000:.0f} ms)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Mode', type=int, default=1, help='Mode 1: one frame 2: n frame')
    args = parser.parse_args()
    
    if args.Mode == 1:
        # Drone
        generate_flying_drone_wav('drone_audio_signal.wav', freq=1000, phase_step=0.1)
        # Pilote
        generate_flying_drone_wav('pilot_radio_signal.wav', freq=2450, phase_step=0.15)
        
    if args.Mode  == 2:
        # Drone
        generate_flying_drone_frame_wav()
