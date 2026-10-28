import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and disable cache during development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def run_server():
    os.chdir(DIRECTORY)
    port = PORT
    server = None
    for attempt_port in range(port, port + 10):
        try:
            server = socketserver.TCPServer(("", attempt_port), Handler)
            port = attempt_port
            break
        except OSError:
            continue

    if not server:
        print("Could not bind to any port between 8000 and 8010.")
        sys.exit(1)

    url = f"http://localhost:{port}/index.html"
    print(f"FreshTrack Food Waste Intelligence Dashboard running at: {url}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run_server()
