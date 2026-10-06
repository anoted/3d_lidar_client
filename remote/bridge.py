#!/usr/bin/env python3
"""LiDAR bridge -- runs ON THE ROBOT (uploaded and launched over SSH by server.py).

Reads point clouds and writes length-prefixed binary frames to stdout, so the
data rides back over the SSH channel (no extra ports, dies with the session).

Sources:
  udp  raw Livox SDK2 point packets (MID-360 pushes them to this host, :56301).
       Needs no driver -- but conflicts with livox_ros_driver2 if that is running.
  ros  any sensor_msgs/PointCloud2 topic (needs a sourced ROS 2 environment).

Frame on stdout:  u32 len | payload
payload (points): b"LDR1" | u32 n | f64 unix_time | i16[n*3] xyz (cm) | u8[n] intensity
payload (status): b"STA1" | utf-8 JSON

Python 3.8 compatible (Ubuntu 20.04 / Foxy).
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

OUT = sys.stdout.buffer
OUT_LOCK = threading.Lock()

HEADER_LEN = 36
# Livox SDK2 point formats: 1 = Cartesian high (mm, int32), 2 = Cartesian low (cm, int16)
PT_HIGH = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("r", "u1"), ("tag", "u1")])
PT_LOW = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("r", "u1"), ("tag", "u1")])


def log(msg):
    sys.stderr.write("[bridge] %s\n" % msg)
    sys.stderr.flush()


def write_payload(payload):
    with OUT_LOCK:
        try:
            OUT.write(struct.pack("<I", len(payload)) + payload)
            OUT.flush()
        except (BrokenPipeError, OSError):
            os._exit(0)


def send_status(**fields):
    write_payload(b"STA1" + json.dumps(fields).encode())


class FrameBuilder:
    """Accumulates points and emits a frame every 1/rate seconds."""

    def __init__(self, rate, max_points):
        self.period = 1.0 / rate
        self.max_points = max_points
        self.xyz = []   # float32 (k,3) arrays in metres
        self.inten = []
        self.last_flush = time.monotonic()
        self.stat_t = self.last_flush
        self.stat_in = 0
        self.stat_frames = 0

    def add(self, xyz_m, inten):
        if len(xyz_m):
            self.xyz.append(xyz_m)
            self.inten.append(inten)
            self.stat_in += len(xyz_m)

    def maybe_flush(self, extra=None):
        now = time.monotonic()
        if now - self.last_flush < self.period:
            return
        self.last_flush = now
        if self.xyz:
            xyz = np.concatenate(self.xyz)
            inten = np.concatenate(self.inten)
            self.xyz, self.inten = [], []
            if len(xyz) > self.max_points:
                idx = np.random.choice(len(xyz), self.max_points, replace=False)
                xyz, inten = xyz[idx], inten[idx]
            q = np.clip(np.round(xyz * 100.0), -32767, 32767).astype("<i2")
            i8 = np.clip(inten, 0, 255).astype(np.uint8)
            write_payload(b"LDR1" + struct.pack("<Id", len(q), time.time()) + q.tobytes() + i8.tobytes())
            self.stat_frames += 1
        if now - self.stat_t >= 1.0:
            dt = now - self.stat_t
            fields = {"points_per_s": int(self.stat_in / dt), "frames_per_s": round(self.stat_frames / dt, 1)}
            if extra:
                fields.update(extra())
            send_status(**fields)
            self.stat_t, self.stat_in, self.stat_frames = now, 0, 0


def run_udp(args, fb):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        send_status(error="cannot bind UDP %s:%d (%s) -- is livox_ros_driver2 running? use --source ros"
                    % (args.bind, args.port, e))
        raise
    sock.settimeout(0.2)
    log("listening for Livox packets on %s:%d" % (args.bind, args.port))
    send_status(source="udp", detail="Livox UDP %s:%d" % (args.bind, args.port))
    packets = [0]
    senders = set()
    gravity = []  # low-passed accelerometer (g units, sensor frame) -> lets the viewer level the cloud
    threading.Thread(target=imu_listener, args=(args, gravity), daemon=True).start()

    def extra():
        n, packets[0] = packets[0], 0
        s = sorted(senders)
        senders.clear()
        fields = {"packets_per_s": n, "lidar": s}
        if gravity:
            fields["gravity"] = [round(float(v), 4) for v in gravity[0]]
        return fields

    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            data = None
        if data and len(data) > HEADER_LEN:
            dot_num = struct.unpack_from("<H", data, 5)[0]
            dtype = {1: PT_HIGH, 2: PT_LOW}.get(data[10])
            if dtype is not None:
                count = min(dot_num, (len(data) - HEADER_LEN) // dtype.itemsize)
                pts = np.frombuffer(data, dtype=dtype, count=count, offset=HEADER_LEN)
                xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
                xyz *= 0.001 if dtype is PT_HIGH else 0.01
                valid = np.any(xyz != 0, axis=1)
                fb.add(xyz[valid], pts["r"][valid])
                packets[0] += 1
                senders.add(addr[0])
        fb.maybe_flush(extra)


def imu_listener(args, gravity):
    """Livox IMU packets: header + 6 float32 (gyro rad/s xyz, accel g xyz)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.bind, args.imu_port))
    except OSError as e:
        log("IMU port %d unavailable (%s); auto-level disabled" % (args.imu_port, e))
        return
    while True:
        data = sock.recv(2048)
        if len(data) >= HEADER_LEN + 24:
            acc = np.array(struct.unpack_from("<3f", data, HEADER_LEN + 12))
            if not gravity:
                gravity.append(acc)
            else:
                gravity[0] = gravity[0] * 0.98 + acc * 0.02


def run_ros(args, fb):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, PointField

    np_types = {PointField.INT8: "i1", PointField.UINT8: "u1", PointField.INT16: "<i2",
                PointField.UINT16: "<u2", PointField.INT32: "<i4", PointField.UINT32: "<u4",
                PointField.FLOAT32: "<f4", PointField.FLOAT64: "<f8"}
    rclpy.init()
    node = rclpy.create_node("lidar_web_bridge")
    lock = threading.Lock()
    msgs = [0]

    def on_cloud(msg):
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in "xyz"):
            return
        iname = next((k for k in ("intensity", "reflectivity", "i") if k in fields), None)
        names = ["x", "y", "z"] + ([iname] if iname else [])
        dt = np.dtype({"names": names,
                       "formats": [np_types[fields[k].datatype] for k in names],
                       "offsets": [fields[k].offset for k in names],
                       "itemsize": msg.point_step})
        pts = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
        valid = np.all(np.isfinite(xyz), axis=1) & np.any(xyz != 0, axis=1)
        inten = pts[iname].astype(np.float32) if iname else np.zeros(len(pts), np.float32)
        with lock:
            fb.add(xyz[valid], inten[valid])
            msgs[0] += 1

    node.create_subscription(PointCloud2, args.topic, on_cloud, qos_profile_sensor_data)
    log("subscribed to %s" % args.topic)
    send_status(source="ros", detail="ROS 2 topic %s" % args.topic)

    def extra():
        n, msgs[0] = msgs[0], 0
        return {"msgs_per_s": n}

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        with lock:
            fb.maybe_flush(extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["udp", "ros"], default="udp")
    ap.add_argument("--bind", default="0.0.0.0", help="udp: address to bind")
    ap.add_argument("--port", type=int, default=56301, help="udp: Livox point-data port")
    ap.add_argument("--imu-port", type=int, default=56401, help="udp: Livox IMU port")
    ap.add_argument("--topic", default="/livox/lidar", help="ros: PointCloud2 topic")
    ap.add_argument("--rate", type=float, default=10.0, help="frames per second sent")
    ap.add_argument("--max-points", type=int, default=25000, help="per frame, random subsample above this")
    args = ap.parse_args()

    # Exit as soon as the SSH session goes away (stdin hits EOF).
    threading.Thread(target=lambda: (sys.stdin.read(), os._exit(0)), daemon=True).start()

    fb = FrameBuilder(args.rate, args.max_points)
    try:
        (run_udp if args.source == "udp" else run_ros)(args, fb)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
