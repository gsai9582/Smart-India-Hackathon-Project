import asyncio
from bleak import BleakClient, BleakScanner
import joblib
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

DEVICE_NAME = "SmartBat"
MODEL_PATH = "../model/bat_model.pkl"

# scoring state
round_times = deque(maxlen=100)
player_score = deque(maxlen=100)
opponent_score = deque(maxlen=100)

current_player_score = 0
current_opponent_score = 0

# model and helper
model = joblib.load(MODEL_PATH)
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "piezo"]

# queue for data from BLE → main thread
from queue import Queue
packet_queue = Queue()


async def find_device():
    print("Scanning for device...")
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == DEVICE_NAME:
            print("Found:", d)
            return d
    raise RuntimeError("SmartBat not found")


async def ble_task():
    device = await find_device()
    async with BleakClient(device) as client:
        print("Connected to", DEVICE_NAME)

        # choose first notify characteristic
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties:
                    data_char = char
                    break

        print("Using characteristic:", data_char.uuid)

        def handler(_, data: bytearray):
            text = data.decode("utf-8").strip()
            packet_queue.put((datetime.now(), text))

        await client.start_notify(data_char, handler)
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await client.stop_notify(data_char)


def process_packet(timestamp, text):
    global current_player_score, current_opponent_score

    try:
        ax, ay, az, gx, gy, gz, piezo, t_ms = text.split(",")
        values = np.array([[float(ax), float(ay), float(az),
                            float(gx), float(gy), float(gz),
                            float(piezo)]])
    except ValueError:
        # corrupted packet
        return

    label = model.predict(values)[0]

    # --- SCORING LOGIC (simple) ---
    if label == "strike":
        current_player_score += 1
    elif label == "miss":
        current_opponent_score += 1
    # "serve" or "none" → no score change

    round_times.append(timestamp)
    player_score.append(current_player_score)
    opponent_score.append(current_opponent_score)


# --- Matplotlib live plot setup ---

plt.style.use("default")
fig, ax = plt.subplots()
line1, = ax.plot([], [], marker="o", label="Player")
line2, = ax.plot([], [], marker="o", label="Opponent")
ax.set_xlabel("Time")
ax.set_ylabel("Score")
ax.set_title("Live Smart Bat Scoring")
ax.legend()
fig.autofmt_xdate()

def update_plot(frame):
    # consume all packets currently in queue
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

    return line1, line2

ani = FuncAnimation(fig, update_plot, interval=200)

def run_dashboard():
    # start BLE asyncio in a background thread
    import threading
    loop = asyncio.new_event_loop()

    def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ble_task())

    t = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    t.start()

    # start the matplotlib event loop (main thread)
    print("Starting dashboard...")
    plt.show()

if __name__ == "__main__":
    run_dashboard()
