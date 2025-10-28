Drone 3D Localization

## Drone & Pilot Localization

This project estimates and visualizes the 3D positions of both a **drone** and its **pilot** based on signals captured by a spherical microphone array. 
Leveraging sound (drone motor noise) and radio (pilot transmitter) localization, it demonstrates advanced 3D source localization using time difference of arrival (TDOA) and SRP-PHAT beamforming techniques.

### Features

- **Spherical Microphone Array Simulation**: The system simulates (or reads from file) multi-microphone signal recordings arranged in a spherical geometry.
- **Dual Source Localization**: Simultaneous detection and 3D localization of:
  - The drone (using acoustic signals, e.g., propeller/motor noise)
  - The pilot (using a modulated radio signal from the remote control)
- **Beamforming with SRP-PHAT**: Applies Steered Response Power Phase Transform (SRP-PHAT) to estimate source directions robustly in the presence of noise.
- **3D Visualization**: Plots microphones, drone, and pilot positions in an interactive 3D matplotlib scene.
- **Simulation Mode**: Includes signal generators to simulate both drone audio and radio pilot signals if no real data is available.

### How It Works ?

1. **Signal Loading**: The program loads or simulates two multi-channel WAV files: one for the drone, one for the pilot (`drone_audio_signal.wav`, `pilot_radio_signal.wav`). Each channel represents a different microphone.
2. **Bandpass Filtering**: Applies frequency filtering appropriate to the expected signal of each source (e.g., 1kHz–1.1kHz for drone propellers, 2.4–2.5kHz for remote controls).
3. **Direction Estimation**: For each source, the program computes the most probable direction of arrival using beamforming algorithms.
4. **3D Position Calculation**: Converts the angle estimates to 3D Cartesian coordinates with respect to the microphone array center.
5. **Scene Plotting**: Displays the computed positions—microphones (blue), drone (red star), and pilot (green triangle)—in a 3D plot for intuitive interpretation.

