# LiDAR Live — web viewer for the G1's MID-360

Live 3D point cloud from the robot (`unitree@192.168.1.21`) in your browser.

```bash
pip install paramiko websockets      # only local deps
python3 server.py                    # then open http://localhost:8080
```

## How it works

```
 MID-360 ──UDP:56301/56401──▶ Jetson (192.168.1.21)            this PC
 192.168.123.120               remote/bridge.py ──SSH stdout──▶ server.py ──WebSocket──▶ browser (three.js)
```

- `server.py` SSHes into the robot, uploads `remote/bridge.py` to `~/.lidar_web/`, and runs it.
  Frames come back over the SSH channel itself, so no ports are opened on the robot.
  The bridge exits when the SSH session closes. If the link drops, the server reconnects every 3 s.
- The bridge batches points into 10 Hz frames, subsampled to 25k points max and quantized to int16 cm,
  about 0.8 MB/s. It also forwards the sensor's accelerometer, and the viewer uses it to level the cloud.
  The G1's lidar is mounted upside-down and pitched about 35°.

## Sources

| `--source` | What it reads | When to use |
|---|---|---|
| `udp` (default) | Raw Livox SDK2 packets the MID-360 already pushes to the Jetson | Works now, no driver needed |
| `ros` | Any `sensor_msgs/PointCloud2` topic (Foxy + CycloneDDS env is sourced automatically) | When a ROS 2 driver is running |

```bash
python3 server.py --source ros --topic /livox/lidar
```

If `livox_ros_driver2` is running, it owns UDP port 56301, so use `--source ros`. Launch the driver with
`xfer_format: 0` so it publishes `PointCloud2` rather than Livox `CustomMsg`.

## Options

`--host --user --password` (or env `LIDAR_HOST`, `LIDAR_USER`, `LIDAR_PASSWORD`), `--rate` (frames/s),
`--max-points` (per frame), `--port` (web port, default 8080). The server listens on `0.0.0.0`,
so other machines on the LAN can open `http://<this-pc>:8080` too.

## Viewer

Color by height, intensity or distance (auto-ranged), or a solid color. You can also set point size, trail length
(how many seconds of points stay on screen) and orientation (IMU-level, raw sensor frame, or flipped).
There are also a grid at the detected floor height, pause (Space), top view, and reset view (R).

**Measure (M):** click two points and a line appears with its length in meters and inches. The label also shows the horizontal (↔) and
vertical (↕) parts, measured in the leveled frame. Clicks snap to the nearest visible lidar point within 10 px.
Pause first (Space) for precise picks. Esc cancels, and *Clear lines* removes all lines.
