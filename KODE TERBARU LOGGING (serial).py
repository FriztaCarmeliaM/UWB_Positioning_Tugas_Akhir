import serial
import struct
import csv
from datetime import datetime
import math
import time
import threading

START_TIME = time.perf_counter()

UWB_PORT = "/dev/ttyUSB0"
UWB_BAUD = 115200
CSV_FILE = "uwb_data2.csv"
HEIGHT_DIFF = 1.11  # meter
SAMPLING_TIME = 0.05 # in seconds (0.05 = 20 Hz)

uwb_distances = {}   # address -> distance
uwb_update_time = {} # address -> waktu update

# Lock to ensure thread-safe read/write on the dictionaries
data_lock = threading.Lock()

def timenow():
    return time.perf_counter() - START_TIME

def read_serial_data():
    """Background thread function to constantly read and parse serial data."""
    ser = serial.Serial(UWB_PORT, UWB_BAUD, timeout=1)
    buffer = b''
    FRAME_SIZE = 6
    FOOTER = b'\xFF\xFF'

    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            
            buffer += data

            while len(buffer) >= FRAME_SIZE:
                if buffer[4:6] == FOOTER:
                    frame = buffer[:6]
                    buffer = buffer[6:]

                    address, distance = struct.unpack('<HH', frame[:4])

                    # Acquire lock before updating shared dictionaries
                    with data_lock:
                        uwb_distances[address] = distance / 100.0  # meter
                        uwb_update_time[address] = timenow()
                else:
                    buffer = buffer[1:]
    except Exception as e:
        print(f"Serial Error: {e}")
    finally:
        if ser.is_open:
            ser.close()

def save_to_csv():
    """Reads the current state of variables and saves them to CSV."""

    current_time = timenow()

    # Acquire lock to safely read dictionaries
    with data_lock:
        # pastikan 3 anchor tersedia
        if not all(k in uwb_distances for k in [0x81, 0x82, 0x83]):
            # Silenced the error so it doesn't spam your terminal while waiting for anchors
            return

        d2 = uwb_distances[0x82]
        d3 = uwb_distances[0x83]
        d1 = uwb_distances[0x81]

        t1 = current_time-uwb_update_time[0x81]
        t2 = current_time-uwb_update_time[0x82]
        t3 = current_time-uwb_update_time[0x83]

    def estimated_l(input_dist):
        return math.sqrt(max(0, input_dist**2 - HEIGHT_DIFF**2))
    # Time formatting and estimated length happens outside the lock to keep the lock duration as short as possible
    el1 = estimated_l(uwb_distances[0x81])
    el2 = estimated_l(uwb_distances[0x82])
    el3 = estimated_l(uwb_distances[0x83])
    row = [current_time, d1, d2, d3, t1, t2, t3, el1, el2, el3]

    print(row)

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)        

def logging_loop():
    """Main thread loop that fires based on SAMPLING_TIME."""
    while True:
        loop_start = time.perf_counter()
        
        save_to_csv()
        
        # Calculate how long the save took, and sleep for the remainder of SAMPLING_TIME
        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0, SAMPLING_TIME - elapsed)
        time.sleep(sleep_time)

def main():
    # Initialize CSV Headers
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "d1","d2","d3",
            "t_uwb_1","t_uwb_2","t_uwb_3",
            "el1","el2","el3"
        ])

    # Start the serial reading in a background thread (daemon=True means it closes when main exits)
    serial_thread = threading.Thread(target=read_serial_data, daemon=True)
    serial_thread.start()

    print(f"Listening on {UWB_PORT} | Sampling every {SAMPLING_TIME} seconds...")

    # Run the timed logging loop in the main thread
    try:
        logging_loop()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

if __name__=="__main__":
    main()
