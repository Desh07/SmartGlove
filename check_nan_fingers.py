import serial
import time
import argparse

def check_nan_fingers(port, baudrate, duration=10):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baud...")
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        return

    start_time = time.time()
    nan_counts = {}

    print(f"Checking for NaN fingers for {duration} seconds...")
    while time.time() - start_time < duration:
        try:
            # Read a line from serial and decode
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("DATA"):
                parts = line.split(',')
                # Skip header lines that look like "DATA,P36,P39..."
                if len(parts) > 1 and parts[1].upper().startswith('P'):
                    continue
                
                # Check sensor values for 'nan'
                for i, val in enumerate(parts[1:]):
                    if val.lower() == 'nan':
                        finger_idx = i + 1
                        nan_counts[finger_idx] = nan_counts.get(finger_idx, 0) + 1
                        
        except Exception as e:
            pass
            
    # Print Results
    print("\n" + "="*30)
    if nan_counts:
        print("⚠️ NaN Fingers Detected!")
        print("="*30)
        for finger, count in sorted(nan_counts.items()):
            print(f"Finger {finger}: returned NaN {count} times")
    else:
        print("✅ No NaN fingers detected.")
        print("All active sensors are working properly.")
    print("="*30)
        
    ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ESP32 flex sensor serial stream for NaN values.")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate")
    parser.add_argument("--duration", type=int, default=10, help="Duration to scan in seconds")
    
    args = parser.parse_args()
    check_nan_fingers(args.port, args.baudrate, args.duration)
