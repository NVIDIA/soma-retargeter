"""Regression: disconnect during computation must not close a live reply handle."""
from multiprocessing import Pipe
import queue
import threading
if __package__:
    from .geometry_transport import read_requests, send_response
else:
    from geometry_transport import read_requests, send_response

def main():
    incoming=queue.Queue()
    for _ in range(100):
        server,client=Pipe(duplex=True)
        reader=threading.Thread(target=read_requests,args=(server,incoming),daemon=True)
        reader.start();client.send(dict(operation='work'));client.close()
        conn,request=incoming.get(timeout=3)
        marker_conn,marker=incoming.get(timeout=3)
        reader.join(timeout=3)
        assert conn is server is marker_conn and request==dict(operation='work') and marker is None
        assert not server.closed,'Reader closed handle while service still owns a request'
        assert send_response(server,dict(result='finished')) is False
        marker_conn.close()
        # A following client must still be serviced normally.
        server,client=Pipe(duplex=True)
        assert send_response(server,dict(result=42))
        assert client.recv()==dict(result=42)
        server.close();client.close()
    print('PASS: 100 disconnect-during-compute cycles and following live replies')

if __name__=='__main__':main()
