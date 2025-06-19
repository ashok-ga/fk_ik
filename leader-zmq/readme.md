
# 🦾 Dynamixel Fast Telemetry Publisher (C++) + ZMQ Subscriber (Python)

This project reads **present position** from multiple Dynamixel motors (IDs 0–6) using `GroupFastBulkRead` at **4 Mbps**, and publishes the data over **ZeroMQ PUB** socket. A Python ZeroMQ subscriber listens and processes the data.

---

## 📁 Project Structure

```
leader-zmq/
├── dxl_pub                
├── leader_pub.cpp       # C++ publisher using GroupFastBulkRead + ZMQ
├── leader_sub.py        # Python subscriber that computes FPS
├── group_read             
├── read_dxl.cpp
├── read_dxl.py
```

---

## 🛠️ Requirements

### ✅ C++ Side
- Dynamixel SDK (C++)
- ZeroMQ C++ (`libzmq3-dev`)
- g++

```bash
sudo apt install libzmq3-dev
git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
cd DynamixelSDK/cpp && mkdir build && cd build
cmake .. && make -j && sudo make install
```

### ✅ Python Side
```bash
pip install pyzmq
```

---

## ⚙️ Build & Run C++ Publisher

### 🔹 Build
```bash
g++ leader_pub.cpp -o dxl_pub -ldxl_x64_cpp -lzmq
```

> Make sure `dynamixel_sdk/` is in your `/usr/local/include/`

### 🔹 Run
```bash
./dxl_pub
```

This will:
- Open `/dev/ttyUSB0` at `4,000,000` baud
- Continuously read present position from motors 0–6
- Publish a JSON message like:

```json
{"id0":12345,"id1":12401,"id2":12678,"id3":12023,"id4":12511,"id5":12789,"id6":12222}
```

---

## 📥 Python Subscriber

### 🔹 `leader_sub.py`

```python
import zmq
import time

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://localhost:5556")
sock.setsockopt_string(zmq.SUBSCRIBE, '')

count = 0
start = time.time()

while True:
    sock.recv_string()
    count += 1
    if count % 500 == 0:
        elapsed = time.time() - start
        print(f"FPS: {count / elapsed:.2f}")
```

### 🔹 Run It
```bash
python3 leader_sub.py
```

---

## ✅ Result

- You will see **FPS updates** every 500 frames on the Python side.
- This lets you benchmark the maximum pub/sub performance of your system using real Dynamixel data.

---

## 🚀 Optional Improvements

- Add timestamps to messages
- Use `msgpack` or `protobuf` instead of raw JSON
- Extend to include velocity, current, or effort
- Add logging or real-time visualization in Python