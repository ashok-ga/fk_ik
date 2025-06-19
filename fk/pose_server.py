# pose_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json

latest_pose = None

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global latest_pose
        length = int(self.headers.get('content-length'))
        data = self.rfile.read(length)
        latest_pose = json.loads(data)
        self.send_response(200)
        self.end_headers()

def run_server(port=9001):
    server = HTTPServer(('localhost', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Pose server running on port {port}")

def get_latest_pose():
    global latest_pose
    return latest_pose
