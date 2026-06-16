import argparse
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def cache_key(model, mode, image_path, prompt):

    image = Path(
        image_path
    ).resolve()
    stat = image.stat()
    digest = hashlib.sha256()

    for value in [
        str(
            Path(
                model
            ).resolve()
        ),
        mode,
        str(
            image
        ),
        str(
            stat.st_size
        ),
        str(
            stat.st_mtime_ns
        ),
        prompt
    ]:
        digest.update(
            value.encode(
                "utf-8"
            )
        )
        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def run_cli(config, image_path, prompt):

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False
    ) as output_file:
        output_path = Path(
            output_file.name
        )

    cmd = [
        config["binary"],
        "detect",
        "--model",
        config["model"],
        "--input",
        image_path,
        "--prompt",
        prompt,
        "--mode",
        config["mode"],
        "--output",
        str(
            output_path
        )
    ]

    if config.get(
        "threads"
    ):
        cmd.extend([
            "--threads",
            str(
                config["threads"]
            )
        ])

    try:
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or f"{config['binary']} exited with {result.returncode}"
            )

        payload = json.loads(
            output_path.read_text()
        )
        payload["_wrapper"] = {
            "cached":
                False,
            "seconds":
                elapsed,
            "mode":
                config["mode"]
        }

        return payload
    finally:
        output_path.unlink(
            missing_ok=True
        )


class LocateAnythingHandler(BaseHTTPRequestHandler):

    server_version = "LocateAnythingCppWrapper/1.0"

    def do_GET(self):

        if self.path != "/health":
            self.send_error(
                404
            )
            return

        self.write_json({
            "ok":
                True,
            "mode":
                self.server.config["mode"],
            "model":
                self.server.config["model"]
        })

    def do_POST(self):

        if self.path != "/detect":
            self.send_error(
                404
            )
            return

        try:
            payload = self.read_json()
            image_path = str(
                payload["image_path"]
            )
            prompt = str(
                payload["prompt"]
            )
            bypass_cache = bool(
                payload.get(
                    "bypass_cache",
                    False
                )
            )

            if not Path(
                image_path
            ).exists():
                raise RuntimeError(
                    f"image_path does not exist: {image_path}"
                )

            key = cache_key(
                self.server.config["model"],
                self.server.config["mode"],
                image_path,
                prompt
            )
            cache_path = self.server.cache_dir / f"{key}.json"

            if cache_path.exists() and not bypass_cache:
                cached = json.loads(
                    cache_path.read_text()
                )
                cached.setdefault(
                    "_wrapper",
                    {}
                )
                cached["_wrapper"]["cached"] = True
                self.write_json(
                    cached
                )
                return

            result = run_cli(
                self.server.config,
                image_path,
                prompt
            )
            result.setdefault(
                "_wrapper",
                {}
            )
            result["_wrapper"]["bypass_cache"] = bypass_cache
            cache_path.write_text(
                json.dumps(
                    result,
                    indent=2
                )
            )
            self.write_json(
                result
            )
        except Exception as exc:
            self.send_response(
                500
            )
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "error":
                        f"{type(exc).__name__}: {exc}"
                }).encode(
                    "utf-8"
                )
            )

    def read_json(self):

        length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        return json.loads(
            self.rfile.read(
                length
            ).decode(
                "utf-8"
            )
        )

    def write_json(self, payload):

        data = json.dumps(
            payload
        ).encode(
            "utf-8"
        )
        self.send_response(
            200
        )
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header(
            "Content-Length",
            str(
                len(
                    data
                )
            )
        )
        self.end_headers()
        self.wfile.write(
            data
        )

    def log_message(self, fmt, *args):

        print(
            f"{self.address_string()} - {fmt % args}",
            flush=True
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8188
    )
    parser.add_argument(
        "--bin",
        required=True
    )
    parser.add_argument(
        "--model",
        required=True
    )
    parser.add_argument(
        "--mode",
        default="fast"
    )
    parser.add_argument(
        "--threads"
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/locate_anything_cpp"
    )
    args = parser.parse_args()

    cache_dir = Path(
        args.cache_dir
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    server = ThreadingHTTPServer(
        (
            args.host,
            args.port
        ),
        LocateAnythingHandler
    )
    server.config = {
        "binary":
            args.bin,
        "model":
            args.model,
        "mode":
            args.mode,
        "threads":
            args.threads
    }
    server.cache_dir = cache_dir

    print(
        f"locate-anything.cpp wrapper listening on http://{args.host}:{args.port}",
        flush=True
    )
    print(
        f"mode={args.mode} cache_dir={cache_dir}",
        flush=True
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
