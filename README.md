
# Real-Time PPG Signal Processing and Heart Rate Analysis

This project captures and processes PPG signals using a MAX30102 sensor for real-time heart rate analysis. It also visualizes data and transmits metrics to a connected client via Bluetooth.

## Features
- **PPG Signal Acquisition**: Real-time data collection using the MAX30102 sensor.
- **Signal Processing**: Preprocessing, quality control, and heart rate calculation using the GODA_pyPPG library.
- **Bluetooth Communication**: Sends processed data to connected clients.
- **Data Visualization**: Displays PPG signals in time and frequency domains.

---

## System Requirements

### Hardware
- Raspberry Pi (or similar microcontroller with Bluetooth support)
- MAX30102 Pulse Oximeter and Heart Rate Sensor
- Bluetooth Module (e.g., integrated Raspberry Pi Bluetooth)

### Software
- **OS**: Raspbian or similar Linux distribution.
- **Python**: Version 3.7 or higher.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yoppman/PPG-heart-rate.git
   cd PPG-heart-rate
   ```

2. Set up a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system packages for Bluetooth communication:
   ```bash
   sudo apt install bluez python3-dbus
   ```

---

## Usage

1. Connect the MAX30102 sensor to your device:
   - Ensure I2C is enabled on your Raspberry Pi.
   - Wire the MAX30102 to the appropriate GPIO pins.

2. Run the script:
   ```bash
   python main.py
   ```

3. Pair your Bluetooth device with the Raspberry Pi and connect to receive data.

4. Visualize the output:
   - Time-domain PPG signal.
   - Frequency-domain spectrum with normalized magnitudes.

---

## Dependencies

- `pyPPG`: [GODA_pyPPG](https://github.com/godamartonaron/GODA_pyPPG)
- Python libraries:
  - `numpy`, `scipy`, `matplotlib`: Data processing and visualization.
  - `pydbus`, `GLib`: Bluetooth communication.
  - `smbus2`: I2C communication.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Key Metrics
- **Heart Rate (BPM)**: Average beats per minute calculated from valid PPG data.
- **RMSSD**: Root mean square of successive differences, an indicator of heart rate variability.
- **HRSTD**: Standard deviation of heart rate over a given window.

---

## File Structure
```
├── main.py                # Main script for sensor control and data visualization
├── calculate_hr.py        # Heart rate calculation and signal processing
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```