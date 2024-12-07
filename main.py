# main.py

import time
import numpy as np
from collections import deque
from smbus2 import SMBus
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json
import socket
from pydbus import SystemBus
from gi.repository import GLib

# Import the calculate_heart_rate function
from calculate_hr import calculate_heart_rate, high_pass_filter


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# MAX30102 Register Addresses
MAX30102_ADDRESS = 0x57
REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
                                 
# **Global Switch for Sending Data**
SEND_DATA = True  # Set to False to disable data sending

def setup_bluetooth_server():
    """
    Set up a Bluetooth server using pydbus and BlueZ.
    """
    bus = SystemBus()
    adapter = bus.get("org.bluez", "/org/bluez/hci0")
    
    # Make the device discoverable and pairable
    adapter_alias = "RaspberryPi"
    adapter.Set("org.bluez.Adapter1", "Alias", GLib.Variant("s", adapter_alias))
    adapter.Set("org.bluez.Adapter1", "Powered", GLib.Variant('b', True))
    adapter.Set("org.bluez.Adapter1", "Discoverable", GLib.Variant('b', True))
    adapter.Set("org.bluez.Adapter1", "Pairable", GLib.Variant('b', True))
    adapter.Set("org.bluez.Adapter1", "DiscoverableTimeout", GLib.Variant('u', 0))  # 0 means unlimited
    
    print("Bluetooth is discoverable and pairable.")
    
    # Create an RFCOMM socket
    server_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_socket.bind(("2C:CF:67:03:0A:C9", 1))  # Empty address, port 1
    server_socket.listen(1)
    print("Waiting for a Bluetooth connection...")
    
    client_socket, client_address = server_socket.accept()
    print(f"Connected to {client_address}")
    
    return server_socket, client_socket

def send_data(client_socket, avg_hr, preprocessed_ppg, xf, yf, IPM, rmssd, hrstd):
    """
    Send data to the connected client via Bluetooth.
    """

    # Normalize yf to range [0, 1] or any fixed range
    yf_min = np.min(yf)
    yf_max = np.max(yf)
    normalized_yf = (yf - yf_min) / (yf_max - yf_min)  # Scale to [0, 1]

    # Optional: Scale to a custom range [a, b]
    target_min, target_max = 0, 10000  # Change to desired range
    normalized_yf = normalized_yf * (target_max - target_min) + target_min

    # normalized_yf = np.full(36, 10000)
    try:
        # Prepare data to send
        data = {
            "avg_hr": avg_hr,
            "data": preprocessed_ppg.tolist(),  # Convert numpy array to list
            "freq_x": xf.tolist(),
            "freq_y": normalized_yf.tolist(),  # Use normalized yf
            "IPM" : IPM,
            "rmssd": rmssd,
            "hrstd": hrstd
        }

        print("Normalized yf:", yf)

        message = json.dumps(data)  # Serialize data as JSON
        client_socket.send(message.encode('utf-8'))  # Send data to client
        print("Data sent to client.")
    except Exception as e:
        print(f"Error sending data: {e}")




class MAX30102:
    def __init__(self, sample_rate, buffer_length, i2c_bus=1):
        self.sample_rate = sample_rate
        self.buffer_length = buffer_length
        self.bus = SMBus(i2c_bus)
        self.address = MAX30102_ADDRESS
        self.buffer_size = sample_rate * buffer_length
        self.ir_buffer = deque(maxlen=self.buffer_size)
        self.setup_sensor()
        
    def setup_sensor(self):
        # Reset
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x40)
        time.sleep(0.1)
        
        # FIFO Configuration
        self.bus.write_byte_data(self.address, REG_FIFO_CONFIG, 0x5F)
        
        # Reset FIFO pointers
        self.bus.write_byte_data(self.address, REG_FIFO_WR_PTR, 0x00)
        self.bus.write_byte_data(self.address, REG_OVF_COUNTER, 0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_RD_PTR, 0x00)
        
        # Mode Configuration - SpO2 mode
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x03)
        
        # SPO2 Configuration
        self.bus.write_byte_data(self.address, REG_SPO2_CONFIG, 0x27)
        
        # LED Configuration
        self.bus.write_byte_data(self.address, REG_LED1_PA, 0x00)  # Disable RED LED
        self.bus.write_byte_data(self.address, REG_LED2_PA, 0x24)  # IR LED
        
        # Enable FIFO interrupts
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_1, 0xC0)

    def read_sensor(self):
        try:
            # Read 6 bytes from FIFO
            data = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)
            
            # Convert bytes to integers
            ir = (data[3] << 16) | (data[4] << 8) | data[5]
            
            # Mask out invalid bits
            ir &= 0x3FFFF
            
            return ir
        except Exception as e:
            print(f"Error reading sensor: {str(e)}")
            return None

def main():
    try:
        # Initialize sensor
        sensor = MAX30102(sample_rate=100, buffer_length=30)
        print("Place your finger on the sensor")
        
        # Set up Bluetooth server if sending data
        if SEND_DATA:
            server_socket, client_socket = setup_bluetooth_server()
        else:
            server_socket = None
            client_socket = None
        
        ir_threshold = 100000
        if len(sensor.ir_buffer) > 1000:
            ir_threshold = np.percentile(sensor.ir_buffer, 10)
        
        # Setup plot with two subplots
        plt.ion()
        fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))
        
        start_time = time.time()
        evaluation_period = 3  # Evaluate heart rate every 3 seconds
        heart_rate_display = None
        window_length = 12  # seconds for heart rate calculation
        window_size = sensor.sample_rate * window_length
    
        while True:
            # Read sensor
            ir = sensor.read_sensor()
            if ir is not None and ir >= ir_threshold:
                sensor.ir_buffer.append(ir)
            else:
                sensor.ir_buffer.append(0)
            
            # Process data every evaluation_period
            if time.time() - start_time >= evaluation_period:
                start_time = time.time()  # Reset timer
                if len(sensor.ir_buffer) >= window_size:
                    ir_data = np.array(sensor.ir_buffer)[-window_size:]
                    

                    # Computationally intensive
                    avg_hr, preprocessed_ppg, IPM, rmssd, hrstd  = calculate_heart_rate(
                        data=ir_data,
                        fs=sensor.sample_rate,
                        sqi_threshold=0.3
                    )

                    preprocessed_ppg = high_pass_filter(preprocessed_ppg, cutoff=0.5, fs=sensor.sample_rate)
                    
                    if avg_hr:
                        heart_rate_display = f"Heart Rate: {avg_hr:.2f} BPM"
                        print(f"Heart Rate: {avg_hr:.2f} BPM")
                    else:
                        heart_rate_display = None  # Could not calculate heart rate
                    
                    # Plot preprocessed PPG signal in time domain
                    time_points = np.linspace(0, len(preprocessed_ppg) / sensor.sample_rate, len(preprocessed_ppg))
                    ax_time.clear()
                    ax_time.plot(time_points, preprocessed_ppg, label="PPG Signal", color="blue")
                    ax_time.set_title("PPG Signal (Time Domain)")
                    ax_time.set_xlabel("Time (seconds)")
                    ax_time.set_ylabel("Amplitude")
                    # ax_time.set_ylim(122000, 133000)  # Set the y-axis range
                    ax_time.legend()
                    ax_time.grid(True)
                    
                    # Display heart rate on the plot
                    if heart_rate_display:
                        ax_time.text(0.1, 0.9, heart_rate_display, transform=ax_time.transAxes, fontsize=14, color="green")
                    
                    # Remove DC component using high-pass filter
                    filtered_ppg = high_pass_filter(preprocessed_ppg, cutoff=0.5, fs=sensor.sample_rate)  # Cutoff at 0.5 Hz
                    # Compute and plot frequency domain (FFT)
                    n = len(filtered_ppg)  # Length of the signal
                    yf = np.fft.fft(filtered_ppg)
                    xf = np.fft.fftfreq(n, 1 / sensor.sample_rate)
                    
                    # Only take the positive frequencies
                    idx = np.where(xf >= 0)
                    xf = xf[idx]
                    yf = np.abs(yf[idx])
                    print(f"x length: {len(xf)}")
                    print(xf[36])
                    print(f"y length: {len(yf)}")
                    # Get the minimum and maximum values
                    min_value = np.min(yf[0:36])
                    max_value = np.max(yf[0:36])

                    print(f"y Minimum value: {min_value}")
                    print(f"y Maximum value: {max_value}")

                    # Normalize yf to range [0, 1] or any fixed range
                    yf_min = np.min(yf)
                    yf_max = np.max(yf)
                    normalized_yf = (yf - yf_min) / (yf_max - yf_min)  # Scale to [0, 1]

                    # Optional: Scale to a custom range [a, b]
                    target_min, target_max = 0, 10000  # Change to desired range
                    normalized_yf = normalized_yf * (target_max - target_min) + target_min
                    
                    center_freq = 1
                    x_range = 4
                    ax_freq.clear()
                    ax_freq.plot(xf, normalized_yf, label="Magnitude Spectrum", color="red")
                    ax_freq.set_xlim(center_freq - x_range / 2, center_freq + x_range / 2)  # Limit frequency axis
                    ax_freq.set_xticks(np.linspace(center_freq - x_range / 2, center_freq + x_range / 2, 5))
                    ax_freq.set_title("PPG Signal (Frequency Domain)")
                    ax_freq.set_xlabel("Frequency (Hz)")
                    ax_freq.set_ylabel("Magnitude")
                    ax_freq.legend()
                    ax_freq.grid(True)

                    # Send data via Bluetooth if enabled
                    if SEND_DATA:
                        try:
                            if preprocessed_ppg is not None and len(preprocessed_ppg) > 0:
                                # Send the last few seconds of data
                                num_of_data = sensor.sample_rate * evaluation_period
                                data_to_send = preprocessed_ppg[-num_of_data:] if len(preprocessed_ppg) >= num_of_data else preprocessed_ppg
                                send_data(client_socket, avg_hr, data_to_send, xf[0:36], normalized_yf[0:36], IPM, rmssd, hrstd)
                        except Exception as e:
                            print(f"Error sending data via Bluetooth: {str(e)}")
                    
                    plt.tight_layout()
                    # plt.savefig(f"python_fig{cnt}.jpeg")
                    # cnt += 1
                    plt.pause(0.001)
                else:
                    print(f"We need {window_size - len(sensor.ir_buffer)} more data to calculate heart rate")
                
            time.sleep(0.01)  # Small delay
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'sensor' in locals():
            sensor.bus.close()
        if SEND_DATA:
            if 'client_socket' in locals() and client_socket:
                client_socket.close()
            if 'server_socket' in locals() and server_socket:
                server_socket.close()
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main()