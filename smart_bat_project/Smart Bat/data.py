import asyncio
from bleak import BleakClient, BleakScanner
import csv
from datetime import datetime

DEVICE_NAME = "SmartBat"
CSV_FILE = "training_data.csv"

# global current label (set via keyboard)
current_label = "none"

# --- keyboard listener (simple, terminal-based) ---
import threading

def label_input():
    global current_label
    print("Label keys: [0]=none, [1]=strike, [2]=miss, [3]=serve")
    while True:
        key = input().strip()
        if key in {"0", "1", "2", "3"}:
            mapping = {"0": "none", "1": "strike", "2": "miss", "3": "serve"}
            current_label = mapping[key]
            print(f"Current label: {current_label}")

# --- BLE part ---

async def find_device():
    print("Scanning for device...")
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == DEVICE_NAME:
            print("Found:", d)
            return d
    raise RuntimeError("SmartBat not found")

async def main():
    # start keyboard thread
    threading.Thread(target=label_input, daemon=True).start()

    device = await find_device()
    async with BleakClient(device) as client:
        print("Connected to", DEVICE_NAME)

        # find first notify characteristic
        # (you can also hardcode UUIDs)
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    data_char = char
                    break

        print("Using characteristic:", data_char.uuid)

        # open CSV and write header once
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            # header only if file is empty
            if f.tell() == 0:
                writer.writerow([
                    "timestamp_pc", "ax", "ay", "az",
                    "gx", "gy", "gz", "piezo",
                    "t_ms_bat", "label"
                ])

            def notification_handler(_, data: bytearray):
                line = data.decode("utf-8").strip()
                ax, ay, az, gx, gy, gz, piezo, t_ms = line.split(",")

                writer.writerow([
                    datetime.now().isoformat(),
                    ax, ay, az, gx, gy, gz, piezo, t_ms,
                    current_label
                ])

            await client.start_notify(data_char, notification_handler)
            print("Logging... Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("Stopping...")
            await client.stop_notify(data_char)

if __name__ == "__main__":
    asyncio.run(main())
