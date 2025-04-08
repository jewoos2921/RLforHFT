import re
import os
import json
import asyncio
import typing as T
from uuid import uuid4

from protocol import Protocol, HOST, PORT, FileWithId, Occurrences

ENCODING = "ISO-8859-1"
RESULT_FILENAME = "result.json"


class Worker(Protocol):
    def connection_lost(self, exc):
        print("Connection lost")
        asyncio.get_running_loop().stop()

    def process_command(self, command: bytes, data: T.Any):
        if command == b'map':
            self.handle_map_request(data)
        elif command == b'reduce':
            self.handle_reduce_request(data)
        elif command == b'disconnect':
            self.connection_lost(None)
        else:
            print(f"Unknown command: {command}")

    def mapfn(self, filename: str) -> T.Dict[str, T.List[int]]:
        print(f"{filename} processing map of file")
        word_counts: T.Dict[str, T.List[int]] = {}
        with open(filename, 'r', encoding=ENCODING) as f:
            for line in f:
                words = re.split("\W+", line)
                for word in words:
                    word = word.lower()
                    if word != "":

                        if word not in word_counts:
                            word_counts[word] = []
                        word_counts[word].append(1)
        return word_counts

    def combinefn(self, results: T.Dict[str, T.List[int]]) -> Occurrences:
        combined_results: Occurrences = {}
        for key in results.keys():
            combined_results[key] = sum(results[key])
        return combined_results

    def reducefn(self, map_files: T.Dict[str, str]) -> Occurrences:
        reduced_result: Occurrences = {}
        for filename in map_files.values():
            with open(filename, "r") as f:
                print(f"Reducing {filename}")
                d = json.load(f)
                for k, v in d.items():
                    reduced_result[k] = v + reduced_result.get(k, 0)
        return reduced_result

    def handle_map_request(self, map_file: FileWithId):
        print(f"Handling map request {map_file}")
        temp_results = self.mapfn(map_file[1])
        results = self.combinefn(temp_results)
        temp_file = self.save_map_results(results)
        self.send_command(
            command=b"mapdone",
            data=(map_file[0], temp_file)
        )

    def save_map_results(self, results: Occurrences) -> str:
        temp_dir = self.get_temp_dir()
        temp_file = os.path.join(temp_dir, f"{uuid4()}.json")
        print(f"Saving map results to {temp_file}")
        with open(temp_file, "w") as f:
            d = json.dumps(results)
            f.write(d)
        print(f"Saved map results to {temp_file}")
        return temp_file

    def handle_reduce_request(self, map_files: T.Dict[str, str]):

        results = self.reducefn(map_files)

        with open(RESULT_FILENAME, "w") as f:
            d = json.dumps(results)
            f.write(d)
        self.send_command(
            command=b"reducedone",
            data=("0", RESULT_FILENAME)
        )


def main():
    event_loop = asyncio.get_event_loop()
    coro = event_loop.create_connection(Worker, HOST, PORT)
    event_loop.run_until_complete(coro)
    event_loop.run_forever()
    event_loop.close()


if __name__ == '__main__':
    main()
