#!/usr/bin/env python3
"""Live LiDAR web viewer -- runs on THIS PC.

Uploads remote/bridge.py to the robot over SSH, runs it, reads point-cloud
frames back over the SSH channel and relays them to browsers via WebSocket.

    python3 server.py                          # raw Livox UDP on the robot (default)
    python3 server.py --source ros --topic /livox/lidar
    then open http://localhost:8080
"""
import argparse
import asyncio
import json
import mimetypes
import os
import pathlib
import shlex
import struct
import threading
import time

import paramiko
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = pathlib.Path(__file__).resolve().parent
STATIC = ROOT / "static"
BRIDGE = ROOT / "remote" / "bridge.py"
REMOTE_DIR = ".lidar_web"

ROS_ENV = ("source /opt/ros/foxy/setup.bash && source ~/cyclonedds_ws/install/setup.bash && "
           "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml && ")


class Hub:
    """Fans frames out to connected browsers; slow clients drop frames instead of lagging."""

    def __init__(self, loop):
        self.loop = loop
        self.clients = {}  # websocket -> asyncio.Queue
        self.status = {"ssh": "starting"}

    def publish(self, msg):
        for q in self.clients.values():
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    def publish_threadsafe(self, msg):
        self.loop.call_soon_threadsafe(self.publish, msg)

    def set_status(self, **fields):
        def apply():
            self.status.update(fields)
            self.publish(json.dumps(self.status))
        self.loop.call_soon_threadsafe(apply)


def read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("bridge closed the stream")
        buf += chunk
    return buf


def ssh_worker(args, hub):
    """Runs forever in a thread: connect, launch bridge, pump frames; reconnect on failure."""
    bridge_args = ["--source", args.source, "--rate", str(args.rate), "--max-points", str(args.max_points)]
    if args.source == "ros":
        bridge_args += ["--topic", args.topic]
    cmd = "exec /usr/bin/python3 -u ~/%s/bridge.py %s" % (REMOTE_DIR, " ".join(shlex.quote(a) for a in bridge_args))
    if args.source == "ros":
        cmd = "bash -c " + shlex.quote(ROS_ENV + cmd)
    target = "%s@%s" % (args.user, args.host)

    while True:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            hub.set_status(ssh="connecting", target=target, error=None)
            client.connect(args.host, username=args.user, password=args.password, timeout=8)
            client.get_transport().set_keepalive(5)
            sftp = client.open_sftp()
            try:
                sftp.mkdir(REMOTE_DIR)
            except IOError:
                pass
            sftp.put(str(BRIDGE), REMOTE_DIR + "/bridge.py")
            sftp.close()
            # A bridge orphaned by an unclean disconnect would still hold the UDP port.
            client.exec_command("pkill -f '[.]%s/bridge.py'" % REMOTE_DIR[1:])[1].channel.recv_exit_status()

            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.channel.settimeout(10)
            threading.Thread(target=pump_stderr, args=(stderr, hub), daemon=True).start()
            hub.set_status(ssh="connected")
            bridge_status = {}
            print("[server] bridge running on %s: %s" % (target, cmd))
            while True:
                payload = read_exact(stdout, struct.unpack("<I", read_exact(stdout, 4))[0])
                if payload[:4] == b"STA1":
                    bridge_status.update(json.loads(payload[4:].decode()))
                    hub.set_status(bridge=dict(bridge_status))
                else:
                    hub.publish_threadsafe(payload)
        except Exception as e:  # noqa: BLE001 -- any failure means reconnect
            print("[server] link error: %r -- retrying in 3 s" % (e,))
            hub.set_status(ssh="disconnected", bridge={}, error=str(e) or type(e).__name__)
        finally:
            client.close()
        time.sleep(3)


def pump_stderr(stream, hub):
    # Shares the channel's read timeout with stdout, so quiet periods raise TimeoutError.
    while True:
        try:
            line = stream.readline()
        except TimeoutError:
            if stream.channel.closed:
                return
            continue
        if not line:
            return
        line = line.rstrip()
        print("[robot] " + line)
        if "Error" in line or "error" in line:
            hub.set_status(error=line)


def static_response(path):
    rel = "index.html" if path in ("", "/") else path.lstrip("/").split("?")[0]
    f = (STATIC / rel).resolve()
    if STATIC not in f.parents or not f.is_file():
        return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"not found")
    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    if f.suffix == ".js":
        ctype = "text/javascript"
    body = f.read_bytes()
    return Response(200, "OK", Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                                        ("Cache-Control", "no-cache")]), body)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("LIDAR_HOST", "192.168.1.21"))
    ap.add_argument("--user", default=os.environ.get("LIDAR_USER", "unitree"))
    ap.add_argument("--password", default=os.environ.get("LIDAR_PASSWORD", "123"))
    ap.add_argument("--source", choices=["udp", "ros"], default="udp",
                    help="udp: decode raw Livox packets (no driver needed); ros: subscribe to a PointCloud2 topic")
    ap.add_argument("--topic", default="/livox/lidar", help="PointCloud2 topic for --source ros")
    ap.add_argument("--rate", type=float, default=10.0, help="frames/s streamed from the robot")
    ap.add_argument("--max-points", type=int, default=25000, help="max points per frame")
    ap.add_argument("--listen", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    hub = Hub(asyncio.get_running_loop())
    threading.Thread(target=ssh_worker, args=(args, hub), daemon=True).start()

    async def handler(ws):
        q = asyncio.Queue(maxsize=2)
        hub.clients[ws] = q
        try:
            await ws.send(json.dumps(hub.status))
            while True:
                await ws.send(await q.get())
        except Exception:  # noqa: BLE001 -- browser went away
            pass
        finally:
            hub.clients.pop(ws, None)

    def process_request(connection, request):
        if request.path != "/ws":
            return static_response(request.path)
        return None

    async with serve(handler, args.listen, args.port, process_request=process_request,
                     max_size=None, compression=None):
        print("[server] open http://localhost:%d" % args.port)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
