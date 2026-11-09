"""Only the service thread closes connections, after queued requests finish."""

def read_requests(connection, incoming):
    try:
        while True:
            incoming.put((connection, connection.recv()))
    except (EOFError, OSError):
        pass
    finally:
        # Queue order keeps this marker after every request from this reader.
        # Closing in the reader can invalidate a concurrent response's handle.
        incoming.put((connection, None))

def send_response(connection, response):
    try:
        connection.send(response)
        return True
    except (OSError, EOFError):
        connection.close()
        return False
