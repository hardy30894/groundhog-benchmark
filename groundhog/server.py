"""The HTTP server the model and every cache listen with."""

import socketserver
from http.server import ThreadingHTTPServer


class LocalServer(ThreadingHTTPServer):
    """A threading HTTP server on a local address. It skips the host-name lookup HTTPServer
    makes when it binds, which takes half a minute on some machines and is never needed here."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
