import selectors
import socket
import types
import os
import signal
import traceback
import time
import numpy as np
from subprocess import Popen


class AllReduce(object):
    def __init__(self, num_proc, host='127.0.0.1', start_port=25000):
        self.num_proc = num_proc
        self.host = host
        self.start_port = start_port
        self.pids = []

        self.selector = None
    
        # data for communication
        self.own_chunk = None
        self.own_network = None

    def __del__(self):
        print(f"pid : {self.pids}")
        time.sleep(2)
        for pid in self.pids:
            os.kill(pid, signal.SIGINT)

    def spawn_processes(self):
        for i in range(1, self.num_proc):
            sub = Popen(['python', 'distributed_process.py', self.host, str(self.start_port + i)])
            self.pids.append(sub.pid)

    def start_connections(self):
        if self.selector:  # Ensure the previous selector is closed properly
            self.selector.close()
        self.selector = selectors.DefaultSelector()
        for connid in range(1, self.num_proc):
            server_addr = (self.host, self.start_port + connid)
            print(f"Starting connection {connid} to {server_addr}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            try:
                sock.connect_ex(server_addr)
            except BlockingIOError:
                pass
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
            data = types.SimpleNamespace(
                connid=connid,
                msg_total=0,
                recv_total=0,
                messages=[],
                outb=b"",
            )
            self.selector.register(sock, events, data=data)

    # 데이터를 읽고 쓰는 부분
    def service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)
            if recv_data == b'2': # 더미 데이터가 들어오면
                print(f"Received {recv_data} from connection {data.connid}")
                data.recv_total += len(recv_data)
                print(f"Closing connection {data.connid}")
                self.selector.unregister(sock)
                sock.close()

            if recv_data == b'1':
                print(f"[BroadCasting] success, {recv_data} from connection {data.connid}")
                self.selector.unregister(sock)
                sock.close()

            if len(recv_data) > 1:
                print(f"[Scatter] success, {recv_data} from connection {data.connid}")
                self.selector.unregister(sock)
                sock.close()

        if mask & selectors.EVENT_WRITE:
            if not data.outb and data.messages:
                data.outb = data.messages.pop(0)
            if data.outb:
                print(f"Sending {data.outb} to connection {data.connid}")
                sent = sock.send(data.outb)
                data.outb = data.outb[sent:]

    def conmmunication_loop(self):
        try:
            # self.start_connections()
            while True:
                events = self.selector.select(timeout=2)                
                if not events:
                    print("No events. Checking connections...")
                    continue
                for key, mask in events:
                    self.service_connection(key, mask)
        except KeyboardInterrupt:
            print("Caught keyboard interrupt, exiting")

        except OSError:
            # print(traceback.format_exc())
            pass
        except Exception as e:
            print("ERROR occurred")
            print(traceback.format_exc())
        finally:
            print("[Closed] selector is closed")
            # self.selector.close()

    def broadcast(self, data):

        self.start_connections()
        self.own_network = data

        data = data.tobytes()
        protocol = self.make_protocol(data, 0)

        # print("length of selector :", len(self.selector.get_map().values()))

        for key in self.selector.get_map().values():
            # print("key :", key)
            key.data.messages.append(protocol)
            key.data.msg_total = len(protocol)
        
        self.conmmunication_loop()

    def scatter(self, data):
        print("[scatter] started scatter pattern")

        self.start_connections()

        unit = len(data) // self.num_proc
        self.own_chunk = data[0 : unit]

        for key in self.selector.get_map().values():
            # make protocol
            data_ = data[unit:unit+unit].tobytes()
            protocol = self.make_protocol(data_, 1, data[unit:unit+unit].shape)

            #bind bytes data to message
            key.data.messages.append(protocol)
            key.data.msg_total = len(protocol)
            unit += unit

        self.conmmunication_loop()

    def allreduce(self, data):
        pass

    def make_protocol(self, data, type_,size=(4,4)):
        type_ = str(type_).encode()
        self.shape = size
        size_of_shape = str(len(size)).encode()
        shape = ''.join(map(str,size)).encode()
        bytes_data = type_ + size_of_shape + shape + data
        return bytes_data
        

def main():
    # initialize all_reduce class
    all_reduce = AllReduce(3)

    # spawn processes for distributed programming
    all_reduce.spawn_processes()

    # broadcast nueral network
    network = np.random.random((4,4)).astype('float32')
    all_reduce.broadcast(network)

    time.sleep(1)

    # scatter the batch data
    batch = np.random.random((15, 4, 4)).astype('float32')
    all_reduce.scatter(batch)

if __name__ == "__main__":
    main()
