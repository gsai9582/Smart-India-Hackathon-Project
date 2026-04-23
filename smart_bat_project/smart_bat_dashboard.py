#!/usr/bin/env python3
"""
SmartBat live dashboard

Usage:
    # simulated/test mode (no BLE required)
    python smart_bat_dashboard.py --test

    # BLE live mode (requires BLE device named DEVICE_NAME and a working model file)
    python smart_bat_dashboard.py --ble

Notes:
 - If MODEL_PATH doesn't exist, a fallback rule-based "model" is used so you can still run the dashboard.
 - Works on Linux/Windows/macOS but BLE needs platform support and permissions.
"""
import argparse
import asyncio
import threading
import time
from collections import deque
from datetime import datetime
from queue import Queue
import random
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Optional imports (bleak & joblib). If missing and BLE mode requested, program will error with helpful message.
try:
    from bleak import BleakClient, BleakScanner
    bleak_available = True
except Exception:
    bleak_available = False

try:
    import joblib
    joblib_available = True
except Exception:
    joblib_available = False

# ---- Configuration ----
DEVICE_NAME = "SmartBat"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "bat_model.pkl")
# You can change the above MODEL_PATH to point to your real model file.

# scoring state (keeps last N rounds for plotting)
MAXLEN = 200
round_times = deque(maxlen=MAXLEN)
player_score = deque(maxlen=MAXLEN)
opponent_score = deque(maxlen=MAXLEN)

current_player_score = 0
current_opponent_score = 0

# features expected (for reference)
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "piezo"]

# queue for packets passed from BLE / simulator to main thread
packet_queue = Queue()

# --- Model loading / fallback ---
model = None
def load_model(path):
    global model
    if joblib_available and os.path.isfile(path):
        try:
            model = joblib.load(path)
            print(f"Loaded model from: {path}")
            return
        except Exception as e:
            print("Failed to load model (joblib) - will use fallback rule-based model. Error:", e)
    else:
        if not joblib_available:
            print("joblib not installed or not available; using fallback rule-based detector.")
        else:
            print(f"Model not found at {path}; using fallback rule-based detector.")
    # fallback model: simple rule-based classifier function
    class FallbackModel:
        def predict(self, X):
            # X is array-like shape (n_samples, 7)
            preds = []
            for row in X:
                # row: ax,ay,az,gx,gy,gz,piezo
                piezo = float(row[6])
                # If piezo bump is high -> likely strike
                if piezo > 0.5:
                    preds.append("strike")
                # if large gyro but small piezo -> miss (swing but no contact)
                elif abs(float(row[3])) + abs(float(row[4])) + abs(float(row[5])) > 50:
                    preds.append("miss")
                else:
                    preds.append("none")
            return np.array(preds)
    model = FallbackModel()
    print("Using fallback rule-based model.")

# --- Packet processing (parsing, prediction, scoring) ---
def process_packet(timestamp, text):
    """
    text expected: "ax,ay,az,gx,gy,gz,piezo,t_ms"
    where t_ms is timestamp from device or sequence (ignored here)
    """
    global current_player_score, current_opponent_score

    try:
        parts = text.strip().split(",")
        if len(parts) < 7:
            # maybe there's no t_ms; handle both 7 or 8-length
            raise ValueError("packet must have at least 7 comma-separated numeric fields")
        # take first 7 as features
        ax, ay, az, gx, gy, gz, piezo = parts[:7]
        values = np.array([[float(ax), float(ay), float(az),
                            float(gx), float(gy), float(gz),
                            float(piezo)]])
    except Exception:
        # corrupted packet
        return

    label = None
    try:
        label = model.predict(values)[0]
    except Exception as e:
        print("Model predict failed:", e)
        label = "none"

    # Simple scoring logic
    if label == "strike":
        current_player_score += 1
    elif label == "miss":
        current_opponent_score += 1
    # serve/none -> no change

    round_times.append(timestamp)
    player_score.append(current_player_score)
    opponent_score.append(current_opponent_score)

# --- BLE task ---
async def find_device():
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == DEVICE_NAME:
            print("Found device:", d)
            return d
    print("Device not discovered in scan.")
    return None

async def ble_task(loop, stop_event):
    if not bleak_available:
        raise RuntimeError("bleak package not available. Install with: pip install bleak")
    print("Scanning for device...")
    device = await find_device()
    if device is None:
        print(f"SmartBat named '{DEVICE_NAME}' not found. BLE mode cannot continue.")
        return

    async with BleakClient(device) as client:
        print("Connected to", DEVICE_NAME)
        await client.get_services()
        # find first notify characteristic
        data_char = None
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    data_char = char
                    break
            if data_char:
                break

        if not data_char:
            print("No notify characteristic found on device.")
            return
        print("Using characteristic:", data_char.uuid)

        def handler(_, data: bytearray):
            try:
                text = data.decode("utf-8").strip()
            except Exception:
                # non-text payload: ignore
                return
            packet_queue.put((datetime.now(), text))

        await client.start_notify(data_char.uuid, handler)
        print("Started notify. Listening for packets... (press Ctrl+C to stop)")
        try:
            # Wait until stop_event is set
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await client.stop_notify(data_char.uuid)
            except Exception:
                pass
            print("Stopped notify.")

# --- Simulator for test mode (generates fake packets) ---
def simulator_task(stop_event, freq_hz=5):
    """
    Generate plausible accelerometer/gyro/piezo packets and put them into packet_queue.
    freq_hz: how many packets per second
    """
    seq = 0
    print("Simulator started (generating fake packets).")
    while not stop_event.is_set():
        # generate background "none"
        base_ax = random.uniform(-0.2, 0.2)
        base_ay = random.uniform(-0.2, 0.2)
        base_az = 9.8 + random.uniform(-0.2, 0.2)
        r = random.random()
        if r < 0.05:
            # strike event: large piezo and spike gyro
            gx = random.uniform(-100, 100)
            gy = random.uniform(-100, 100)
            gz = random.uniform(-100, 100)
            piezo = random.uniform(0.6, 1.5)
            ax = base_ax + random.uniform(-6, 6)
            ay = base_ay + random.uniform(-6, 6)
            az = base_az + random.uniform(-6, 6)
        elif r < 0.15:
            # miss: swing large gyro but small piezo
            gx = random.uniform(-80, 80)
            gy = random.uniform(-80, 80)
            gz = random.uniform(-80, 80)
            piezo = random.uniform(0.0, 0.4)
            ax = base_ax + random.uniform(-3, 3)
            ay = base_ay + random.uniform(-3, 3)
            az = base_az + random.uniform(-3, 3)
        else:
            # none
            gx = random.uniform(-5, 5)
            gy = random.uniform(-5, 5)
            gz = random.uniform(-5, 5)
            piezo = random.uniform(0.0, 0.2)
            ax = base_ax + random.uniform(-0.1, 0.1)
            ay = base_ay + random.uniform(-0.1, 0.1)
            az = base_az + random.uniform(-0.1, 0.1)

        t_ms = int(time.time() * 1000)
        text = f"{ax:.3f},{ay:.3f},{az:.3f},{gx:.3f},{gy:.3f},{gz:.3f},{piezo:.3f},{t_ms}"
        packet_queue.put((datetime.now(), text))
        seq += 1
        time.sleep(1.0 / freq_hz)
    print("Simulator stopped.")

# --- Matplotlib live plot setup ---
plt.style.use("default")
fig, ax = plt.subplots()
line1, = ax.plot([], [], marker="o", label="Player")
line2, = ax.plot([], [], marker="o", label="Opponent")
ax.set_xlabel("Time")
ax.set_ylabel("Score")
ax.set_title("Live SmartBat Scoring")
ax.legend()
fig.autofmt_xdate()

def update_plot(frame):
    # consume all packets
    while not packet_queue.empty():
        ts, text = packet_queue.get()
        process_packet(ts, text)

    if len(round_times) == 0:
        return line1, line2

    x = list(round_times)
    y1 = list(player_score)
    y2 = list(opponent_score)

    line1.set_data(x, y1)
    line2.set_data(x, y2)

    ax.relim()
    ax.autoscale_view()

    # format x-axis nicely (matplotlib will handle)
    return line1, line2

def start_ble_thread_and_loop(loop, stop_event):
    # run ble_task in event loop (to be created per-thread)
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(ble_task(loop, stop_event))
    except Exception as e:
        print("BLE thread ended with exception:", e)

def run_dashboard(mode="test"):
    """
    mode: 'test' or 'ble'
    """
    load_model(MODEL_PATH)
    stop_event = threading.Event()

    # Start BLE or Simulator in a background thread
    if mode == "ble":
        if not bleak_available:
            print("bleak package is not available. Install with 'pip install bleak'")
            return
        # create a new event loop for BLE thread
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=start_ble_thread_and_loop, args=(loop, stop_event), daemon=True)
        t.start()
    else:
        # start simulator thread
        sim_thread = threading.Thread(target=simulator_task, args=(stop_event, 5), daemon=True)
        sim_thread.start()

    # start matplotlib animation in main thread
    ani = FuncAnimation(fig, update_plot, interval=200)

    try:
        print("Starting dashboard window... (close it to stop)")
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Stopping background threads...")
        stop_event.set()
        time.sleep(0.5)
        # give threads a moment to finish

# --- CLI entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="SmartBat live dashboard (BLE or simulated).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Run in test/simulated mode (no BLE).")
    group.add_argument("--ble", action="store_true", help="Run in BLE live mode (connect to SmartBat).")
    args = parser.parse_args()

    mode = "ble" if args.ble else "test"
    run_dashboard(mode=mode)

if __name__ == "__main__":
    main()
