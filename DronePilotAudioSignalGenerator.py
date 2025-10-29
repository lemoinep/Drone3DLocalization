# Author(s): Dr. Patrick Lemoine
# Drone and Pilot Audio Signal Generator
# Simulates multi-microphone audio signals for acoustic monitoring applications.

import numpy as np
from scipy.io.wavfile import write
import argparse
from typing import Tuple
from pathlib import Path

class AudioSignalGenerator:
    """Main class for multi-microphone audio signal generation."""

    def __init__(self, sample_rate: int = 16000, noise_level: float = 0.02):
        """
        Initialize the audio signal generator.

        Args:
            sample_rate: Sampling rate in Hz.
            noise_level: Amount of noise to add to the signal.
        """
        self.sample_rate = sample_rate
        self.noise_level = noise_level

    def _generate_smooth_trajectory(self, start_theta: float, start_phi: float,
                                   end_theta: float, end_phi: float,
                                   samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a smooth trajectory using cosine interpolation.

        Args:
            start_theta: Initial theta angle (degrees).
            start_phi: Initial phi angle (degrees).
            end_theta: Final theta angle (degrees).
            end_phi: Final phi angle (degrees).
            samples: Number of samples.

        Returns:
            Tuple of theta and phi trajectories.
        """
        t = np.linspace(0, 1, samples)
        theta_traj = start_theta + (end_theta - start_theta) * (1 - np.cos(t * np.pi)) / 2
        phi_traj = start_phi + (end_phi - start_phi) * (1 - np.cos(t * np.pi)) / 2
        return theta_traj, phi_traj

    def _calculate_microphone_positions(self, n_mics: int) -> list:
        """
        Compute the angular positions of microphones in a circular array.

        Args:
            n_mics: Number of microphones.

        Returns:
            List of (theta, phi) positions for each microphone.
        """
        mic_positions = []
        for i in range(n_mics):
            theta = 180 * (i + 0.5) / n_mics  # Inclination
            phi = 360 * (i + 0.5) / n_mics    # Azimuth
            mic_positions.append((theta, phi))
        return mic_positions

    def _apply_windowing(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a Hanning window to the signal to avoid discontinuities.

        Args:
            signal: Input signal.

        Returns:
            Windowed signal.
        """
        return signal * np.hanning(len(signal))

    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add white Gaussian noise to the signal.

        Args:
            signal: Input signal.

        Returns:
            Signal with added noise.
        """
        noise = self.noise_level * np.random.randn(len(signal))
        return signal + noise

    def _normalize_and_convert(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize and convert audio to int16 format for WAV.

        Args:
            audio: Multichannel audio array.

        Returns:
            Normalized audio in int16 format.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val
        else:
            audio_normalized = audio
        return np.int16(audio_normalized * 32767)

class DroneSignalGenerator(AudioSignalGenerator):
    """Specialized generator for drone signals with complex trajectory."""

    def generate_flying_drone_signal(self, filename: str, n_mics: int = 16,
                                     duration_s: float = 2.0, frequency: float = 1000,
                                     theta_path: Tuple[float, float] = (20, 160),
                                     phi_path: Tuple[float, float] = (20, 340)) -> None:
        """
        Generate a simulated drone flight signal with a realistic trajectory.

        Args:
            filename: Output file name.
            n_mics: Number of microphones.
            duration_s: Duration in seconds.
            frequency: Drone frequency in Hz.
            theta_path: Theta trajectory (start, end).
            phi_path: Phi trajectory (start, end).
        """
        samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, samples, endpoint=False)
        audio = np.zeros((samples, n_mics))

        # Generate smooth trajectory
        theta_traj, phi_traj = self._generate_smooth_trajectory(
            theta_path[0], phi_path[0], theta_path[1], phi_path[1], samples
        )

        # Microphone positions
        mic_positions = self._calculate_microphone_positions(n_mics)

        # Generate signal for each microphone
        for i, (mic_theta, mic_phi) in enumerate(mic_positions):
            phase_delay = (np.radians(theta_traj - mic_theta) +
                           np.radians(phi_traj - mic_phi))
            base_signal = np.sin(2 * np.pi * frequency * t + phase_delay)
            base_signal = self._apply_windowing(base_signal)
            base_signal = self._add_noise(base_signal)
            audio[:, i] = base_signal

        # Normalize and save
        audio_int16 = self._normalize_and_convert(audio)
        self._save_audio(filename, audio_int16)

        print(f"Realistic flying drone signal generated: {filename} "
              f"({duration_s:.1f}s, {n_mics} mics, {frequency}Hz)")

    def _save_audio(self, filename: str, audio: np.ndarray) -> None:
        """
        Save audio data to WAV file.

        Args:
            filename: Output file name.
            audio: Audio data.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        write(filename, self.sample_rate, audio)

class SimpleSignalGenerator(AudioSignalGenerator):
    """Generator for basic signals (static drone, pilot radio)."""

    def generate_simple_signal(self, filename: str, n_mics: int = 16,
                              duration_s: float = 0.1, frequency: float = 1000,
                              phase_step: float = 0.1) -> None:
        """
        Generate a basic multi-channel sinusoidal signal.

        Args:
            filename: Output file name.
            n_mics: Number of microphones.
            duration_s: Duration in seconds.
            frequency: Frequency in Hz (1000 for drone, 2450 for pilot radio).
            phase_step: Phase variation between microphones.
        """
        samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, samples, endpoint=False)

        signals = []
        for i in range(n_mics):
            phase_shift = i * phase_step
            base_signal = np.sin(2 * np.pi * frequency * t + phase_shift)
            base_signal = self._apply_windowing(base_signal)
            base_signal = self._add_noise(base_signal)
            signals.append(base_signal)

        audio = np.array(signals).T  # (samples, n_mics)
        audio_int16 = self._normalize_and_convert(audio)
        self._save_audio(filename, audio_int16)
        signal_type = "drone" if frequency < 2000 else "pilot radio"
        print(f"{signal_type.capitalize()} signal generated: {filename} "
              f"(freq={frequency}Hz, {self.sample_rate}Hz, {n_mics} mics, "
              f"{duration_s*1000:.0f}ms)")

    def _save_audio(self, filename: str, audio: np.ndarray) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        write(filename, self.sample_rate, audio)

class AudioConfig:
    """Centralized configuration for audio signal parameters."""

    DRONE_CONFIG = {
        'frequency': 1000,
        'phase_step': 0.1,
        'duration_s': 0.1
    }

    PILOT_RADIO_CONFIG = {
        'frequency': 2450,
        'phase_step': 0.15,
        'duration_s': 0.1
    }

    FLYING_DRONE_CONFIG = {
        'frequency': 1000,
        'duration_s': 2.0,
        'theta_path': (20, 160),
        'phi_path': (20, 340)
    }

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Drone and pilot audio signal generator"
    )
    parser.add_argument(
        '--option',
        type=str,
        default='1',
        choices=['1', '2'],
        help='Generation option (1: basic signals, 2: flying drone)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for audio files'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate (Hz)'
    )
    parser.add_argument(
        '--n-mics',
        type=int,
        default=16,
        help='Number of microphones'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.option == '1':
        # Generate basic signals
        simple_generator = SimpleSignalGenerator(
            sample_rate=args.sample_rate
        )
        # Drone signal
        drone_file = output_dir / 'drone_audio_signal.wav'
        simple_generator.generate_simple_signal(
            str(drone_file),
            n_mics=args.n_mics,
            **AudioConfig.DRONE_CONFIG
        )
        # Pilot radio signal
        pilot_file = output_dir / 'pilot_radio_signal.wav'
        simple_generator.generate_simple_signal(
            str(pilot_file),
            n_mics=args.n_mics,
            **AudioConfig.PILOT_RADIO_CONFIG
        )

    elif args.option == '2':
        # Generate flying drone signal
        drone_generator = DroneSignalGenerator(
            sample_rate=args.sample_rate
        )
        flying_drone_file = output_dir / 'drone_audio_signal_frame.wav'
        drone_generator.generate_flying_drone_signal(
            str(flying_drone_file),
            n_mics=args.n_mics,
            **AudioConfig.FLYING_DRONE_CONFIG
        )

if __name__ == "__main__":
    main()
