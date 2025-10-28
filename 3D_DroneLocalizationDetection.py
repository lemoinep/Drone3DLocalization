# Author(s): Dr. Patrick Lemoine

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile

# --- Spherical to Cartesian conversion ---

def sph_to_cart(r, theta_deg, phi_deg):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# --- Generate microphone positions uniformly on a sphere ---

def generate_mic_positions_on_sphere(n_mics, radius):
    indices = np.arange(0, n_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_mics)
    theta = np.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

# --- Bandpass filter ---

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# --- Load generic source signal (audio or radio) ---

def load_source_signal(file_path, n_mics):
    fs, audio = wavfile.read(file_path)
    if len(audio.shape) == 1:
        audio = np.tile(audio[:, None], (1, n_mics))
    if audio.shape[1] != n_mics:
        raise ValueError("Number of microphones in audio does not match configuration.")
    signals = [audio[:, i] for i in range(n_mics)]
    duration_s = audio.shape[0] / fs
    return signals, fs, duration_s

# --- Beamforming (SRP-PHAT) ---

def srp_phat_bf(signals, mic_positions, fs, grid_theta, grid_phi, c=343.0):
    n_mics = mic_positions.shape[0]
    radius = np.linalg.norm(mic_positions[0])
    sig_length = signals[0].shape[0]
    fft_signals = [np.fft.rfft(sig, n=sig_length) for sig in signals]
    freqs = np.fft.rfftfreq(sig_length, 1. / fs)
    results = []
    for th, ph in itertools.product(grid_theta, grid_phi):
        steer_vec = sph_to_cart(radius * 3, th, ph)
        delays = []
        for mic_pos in mic_positions:
            vec = steer_vec - mic_pos
            tau = np.linalg.norm(vec) / c
            delays.append(tau)
        tau0 = delays[0]
        relative_delays = [tau - tau0 for tau in delays]
        steer_sum = np.zeros_like(fft_signals[0])
        for i in range(n_mics):
            phase_shift = np.exp(-2j * np.pi * freqs * relative_delays[i])
            steer_sum += fft_signals[i] * phase_shift / (np.abs(fft_signals[i]) + 1e-10)
        power = np.abs(np.sum(steer_sum))
        results.append((power, th, ph))
    return results

def find_k_max_sources(bf_results, k):
    sorted_results = sorted(bf_results, key=lambda x: x[0], reverse=True)
    selected = []
    for power, th, ph in sorted_results:
        too_close = False
        for _, t_sel, p_sel in selected:
            distance = np.sqrt((th - t_sel)**2 + (ph - p_sel)**2)
            if distance < 10.0:
                too_close = True
                break
        if not too_close:
            selected.append((power, th, ph))
            if len(selected) == k:
                break
    return selected

# --- Modified 3D plot: show drone and pilot separately ---

def plot_3d_scene_with_drone_and_pilot(mic_positions, drone_cart, pilot_cart, mic_radius, source_display_distance):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = mic_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = mic_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = mic_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3)

    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               color='blue', label='Microphones', s=50)

    # Drone
    ax.scatter(drone_cart[0], drone_cart[1], drone_cart[2], color='red', s=120, marker='*', label='Drone')
    ax.plot([0, drone_cart[0]], [0, drone_cart[1]], [0, drone_cart[2]], color='red', linestyle='dashed')

    # Pilot
    ax.scatter(pilot_cart[0], pilot_cart[1], pilot_cart[2], color='green', s=120, marker='^', label='Pilot')
    ax.plot([0, pilot_cart[0]], [0, pilot_cart[1]], [0, pilot_cart[2]], color='green', linestyle='dashed')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Microphones (blue), Drone (red), Pilot (green)')
    axis_limit = source_display_distance * 1.2
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    ax.legend()
    plt.show()

if __name__ == "__main__":
    n_mics = 16
    mic_radius = 0.15
    grid_theta = np.linspace(0, 180, 45)
    grid_phi = np.linspace(0, 360, 90)
    dist_factor = 20.0  # estimated factor for display

    # Load/define drone signal and pilot (remote control) signal
    drone_signal_path = "drone_audio_signal.wav"
    pilot_signal_path = "pilot_radio_signal.wav"   # File to be provided/simulated

    # Drone sound processing
    signals_drone, fs, duration_s = load_source_signal(drone_signal_path, n_mics)
    # Pilot radio signal processing (assumes same WAV format, otherwise adapt accordingly)
    signals_pilot, fs2, duration_s2 = load_source_signal(pilot_signal_path, n_mics)

    # Optional: filtering (adapt values based on nature: audio vs radio)
    for i in range(n_mics):
        signals_drone[i] = bandpass_filter(signals_drone[i], 900, 1100, fs)
        signals_pilot[i] = bandpass_filter(signals_pilot[i], 2400, 2500, fs)  # typical radio

    mic_positions = generate_mic_positions_on_sphere(n_mics, mic_radius)
    # Drone direction
    bf_results_drone = srp_phat_bf(signals_drone, mic_positions, fs, grid_theta, grid_phi)
    k_best_drone = find_k_max_sources(bf_results_drone, k=1)
    theta_drone, phi_drone = k_best_drone[0][1], k_best_drone[0][2]
    cart_drone = sph_to_cart(dist_factor * mic_radius, theta_drone, phi_drone)
    # Pilot direction
    bf_results_pilot = srp_phat_bf(signals_pilot, mic_positions, fs, grid_theta, grid_phi)
    k_best_pilot = find_k_max_sources(bf_results_pilot, k=1)
    theta_pilot, phi_pilot = k_best_pilot[0][1], k_best_pilot[0][2]
    cart_pilot = sph_to_cart(dist_factor * mic_radius, theta_pilot, phi_pilot)

    print(f"Drone position: theta={theta_drone:.1f}° phi={phi_drone:.1f}° | cartesian = {cart_drone}")
    print(f"Pilot position: theta={theta_pilot:.1f}° phi={phi_pilot:.1f}° | cartesian = {cart_pilot}")

    plot_3d_scene_with_drone_and_pilot(mic_positions, cart_drone, cart_pilot, mic_radius, dist_factor * mic_radius)


