import os
import asyncio
import json
import typing as T

# Protocol, HOST, PORT, FileWithId

FileWithId = T.Tuple[str, str]
Occurrences = T.Dict[str, int]

PORT = 12345
HOST = "127.0.0.1"
TEMP_DIR = "temp"
END_MSG = b"EOF"


class Protocol(asyncio.Protocol):
    def __init__(self):
        super().__init__()
        self.buffer = b""

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        print("Connection made")

    def data_received(self, data: bytes):
        self.buffer += data
        if END_MSG in self.buffer:
            if b":" not in data:
                command, _ = self.buffer.split(END_MSG, 1)
                data = None
            else:
                command, data = self.buffer.split(b":", 1)
                data, self.buffer = data.split(END_MSG, 1)
                data = json.loads(data.decode())
            self.process_command(command, data)

    def process_command(self, command: bytes, data: T.Any):
        raise NotImplementedError

    def get_temp_dir(self) -> str:
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, TEMP_DIR)

    def send_command(self, command, data: FileWithId = None) -> None:
        if data:
            pdata = json.dumps(data).encode()
            self.transport.write(command + b":" + pdata + END_MSG)
        else:
            self.transport.write(command + END_MSG)
