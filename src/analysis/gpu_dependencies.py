"""Install the legacy ESM code with its preserved Transformers dependency."""


def install_legacy_esm(esm_revision: str, transformers_revision: str) -> None:
    """Replace only ESM's stale dependency URL before installing normally."""
    import subprocess
    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        source = Path(directory)
        subprocess.run(["git", "init", "--quiet", directory], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                directory,
                "fetch",
                "--depth=1",
                "https://github.com/Biohub/esm.git",
                esm_revision,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", directory, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            check=True,
        )
        metadata = source / "pyproject.toml"
        original = metadata.read_text()
        old = '"transformers @ git+https://github.com/Biohub/transformers.git@main"'
        new = (
            '"transformers @ git+https://github.com/evolutionaryscale/transformers.git@'
            f'{transformers_revision}"'
        )
        if original.count(old) != 1:
            raise RuntimeError("Expected exactly one legacy Transformers dependency in ESM")
        metadata.write_text(original.replace(old, new))
        subprocess.run([sys.executable, "-m", "pip", "install", directory], check=True)


if __name__ == "__main__":
    import sys

    install_legacy_esm(sys.argv[1], sys.argv[2])
