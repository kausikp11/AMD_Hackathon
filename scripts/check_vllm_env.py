from importlib import metadata
import sys


def version_tuple(version):

    parts = []

    for part in version.split("."):
        number = ""

        for char in part:
            if char.isdigit():
                number += char
            else:
                break

        if not number:
            break

        parts.append(
            int(
                number
            )
        )

    return tuple(
        parts
    )


def package_version(name):

    try:
        return metadata.version(
            name
        )
    except metadata.PackageNotFoundError:
        return None


def fail(message):

    print(
        message,
        file=sys.stderr
    )
    raise SystemExit(
        1
    )


def main():

    hub = package_version(
        "huggingface-hub"
    )

    if hub is not None:
        hub_version = version_tuple(
            hub
        )

        if hub_version < (0, 34) or hub_version >= (1, 0):
            fail(
                "\n".join([
                    "Bad vLLM dependency: transformers requires "
                    "huggingface-hub>=0.34.0,<1.0.",
                    f"Installed huggingface-hub: {hub}",
                    "Fix:",
                    "  python3 -m pip install --force-reinstall "
                    "\"huggingface-hub==0.36.0\" --break-system-packages",
                    "Or run:",
                    "  ./scripts/fix_amd_vllm_env.sh"
                ])
            )

    fastapi = package_version(
        "fastapi"
    )
    starlette = package_version(
        "starlette"
    )
    instrumentator = package_version(
        "prometheus-fastapi-instrumentator"
    )

    if fastapi and version_tuple(fastapi) >= (0, 116):
        fail(
            "\n".join([
                "Bad vLLM web dependency: FastAPI >=0.116 can trigger "
                "prometheus_fastapi_instrumentator route crashes in this "
                "AMD image.",
                f"Installed fastapi: {fastapi}",
                f"Installed starlette: {starlette or 'missing'}",
                f"Installed prometheus-fastapi-instrumentator: "
                f"{instrumentator or 'missing'}",
                "Fix:",
                "  python3 -m pip install --force-reinstall "
                "\"fastapi==0.115.14\" \"starlette==0.46.2\" "
                "\"prometheus-fastapi-instrumentator==7.1.0\" "
                "--break-system-packages",
                "Or run:",
                "  ./scripts/fix_amd_vllm_env.sh"
            ])
        )

    print(
        "vLLM dependency preflight OK"
    )


if __name__ == "__main__":
    main()
