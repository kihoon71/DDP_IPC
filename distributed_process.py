import numpy as np
import socket
import logging
from logging.handlers import RotatingFileHandler
import sys
import traceback

class DistributedProcess(object):
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.network = None
        self.input_datachunk = None
        self.reduction_data = None
        self.logger = None 
        self.set_logger()
    
    def __del__(self):
        self.logger.info("DistributedProcess object is deleted")

    def set_logger(self):
        process_id = self.port - 25000
        self.logger = logging.getLogger(f'subprocess_{process_id}')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("subprocesses.log", maxBytes=10000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def open_socket_server(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, self.port))
                s.listen()
                self.logger.info(f"Socket opened on {self.host}:{self.port}")
                while True:
                    conn, addr = s.accept()
                    with conn:
                        self.logger.info(f"Connected by {addr}")
                        while True:
                            data = conn.recv(1024)
                            self.logger.info(f"Received data: {data}")
                            if not data:
                                break
                            if data[0:1] == b'0':
                                self.network = self.decode_protocol(data[1:])
                                self.logger.info(f"Received network from {addr}, network : \n{self.network}")
                                conn.send(b"1")
                            elif data[0:1] == b'1':
                                self.logger.info(f"Received data chunk from {addr}")
                                self.input_datachunk = self.decode_protocol(data[1:])
                                result_forward = self.forward_datachunk()
                                conn.sendall(result_forward)
                            elif data[0:1] == b'2':
                                self.logger.info(f"Received reduction data from {addr}")
                                self.reduction_data = self.decode_protocol(data[1:])
                                self.logger.info(f"Reduction Data : \n{self.reduction_data}")
                                conn.send(b"1")
                            elif data[0:1] == b'3':
                                s.close()
                                sys.exit()
        except Exception as e:
            self.logger.error(f"Exception: {traceback.format_exc()}")

    def decode_protocol(self, recv_data):
        len_of_shape = int(recv_data[0:1].decode())
        shape = tuple(map(int, recv_data[1:1+len_of_shape].decode()))
        buffer_array_data = np.frombuffer(recv_data[1+len_of_shape:], np.float32)
        buffer_array_data = buffer_array_data.reshape(shape)
        self.logger.info(f"buffer_array_data : \n {buffer_array_data}")
        return buffer_array_data

    def forward_datachunk(self):
        result_forward = np.dot(self.input_datachunk, self.network)
        result_forward = result_forward.astype('float32')
        self.logger.info(f"shape : {result_forward.shape}")
        result_forward = result_forward.tobytes()
        self.logger.info(f"bytes_length : {len(result_forward)}")
        return result_forward

if __name__ == "__main__":
    host, port = sys.argv[1], int(sys.argv[2])
    process = DistributedProcess(host, port)
    process.open_socket_server()
