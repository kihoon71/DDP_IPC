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

        # self.selector = None
        self.selector = selectors.DefaultSelector()

        # data for communication
        self.own_chunk = None
        self.own_network = None
        self.forward_data = []
        self.reduction_result = None

        # shape for the chunk
        self.shape = None

    def __del__(self):
        print(f"pid : {self.pids}")
        self.selector.close()
        for pid in self.pids:
            os.kill(pid, signal.SIGINT)

    def spawn_processes(self):
        for i in range(1, self.num_proc):
            sub = Popen(['python', 'distributed_process.py', self.host, str(self.start_port + i)])
            self.pids.append(sub.pid)

    def start_connections(self):
        
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
                print(f"[Scatter] success, {len(recv_data)} from connection {data.connid}")
                forward_result_data = np.frombuffer(recv_data, dtype='float32')
                forward_result_data = forward_result_data.reshape(self.shape)
                self.forward_data.append(forward_result_data)
                self.selector.unregister(sock)
                sock.close()

            if not recv_data:
                print(f"Closing connection {data.connid}")
                self.selector.unregister(sock)
                sock.close()

        if mask & selectors.EVENT_WRITE:
            if not data.outb and data.messages:
                data.outb = data.messages.pop(0)
            if data.outb:
                print(f"Sending {len(data.outb)} to connection {data.connid}")
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
            print("[info] this communication is done")
            # self.selector.close()

    def broadcast(self, data, type_):

        self.start_connections()
        self.own_network = data

        byte_data = data.tobytes()
        protocol = self.make_protocol(byte_data, type_, size=data.shape)

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


    # 실제 reduction 연산의 대상은 forwardpass 이후의 loss 값이다. 여기서는 간단하게 내적 연산으로 대체한다.
    def reduction_op(self):

        #own data result
        master_forward = np.dot(self.own_chunk, self.own_network)
        master_forward = master_forward.astype('float32')
        self.forward_data.append(master_forward)

        self.reduction_result = np.mean(self.forward_data, axis=0)
        return self.reduction_result

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
    all_reduce.broadcast(network, type_=0)

    time.sleep(1)

    # scatter the batch data
    batch = np.random.random((15, 4, 4)).astype('float32')
    all_reduce.scatter(batch)

    # all_reduce the batchdata
    reduction_loss_value = all_reduce.reduction_op()
    all_reduce.broadcast(reduction_loss_value, type_=2)
    print("[Reduction] result of reduction op is broadcasted")

    time.sleep(2)
    # quit all the process
    quit_signal = np.array([])
    all_reduce.broadcast(quit_signal, type_=3)

    time.sleep(3)

if __name__ == "__main__":
    main()
