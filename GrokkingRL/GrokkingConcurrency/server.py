import os
import glob
import asyncio
from scheduler import Scheduler
from protocol import Protocol, HOST, PORT, FileWithId


class Server(Protocol):
    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def connection_made(self, transport: asyncio.Transport) -> None:
        peername = transport.get_extra_info('peername')
        print(f"Connection from {peername}")
        self.transport = transport
        self.start_new_task()

    def start_new_task(self) -> None:
        command, data = self.scheduler.get_next_task()
        self.send_command(command, data)

    def send_command(self, command: bytes,
                     data: FileWithId = None) -> None:
        if command == b"mapdone":

            self.scheduler.map_done(data)
            self.start_new_task()
        elif command == b"reducedone":
            self.scheduler.reduce_done()
            self.start_new_task()

        else:
            print(f"Wrong command received: {command}")


def main():
    event_loop = asyncio.get_event_loop()

    current_path = os.path.abspath(os.getcwd())
    file_locations = list(glob.glob(f'{current_path}/input_files/*.txt'))
    scheduler = Scheduler(file_locations)
    server = event_loop.create_server(
        lambda: Server(scheduler), HOST, PORT
    )

    server = event_loop.run_until_complete(server)
    print(f"Serving on {server.sockets[0].getsockname()}")
    try:
        event_loop.run_forever()
    finally:
        server.close()
        event_loop.run_until_complete(server.wait_closed())
        event_loop.close()


if __name__ == '__main__':
    main()
