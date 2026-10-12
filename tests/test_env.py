import os

from jevbench.env import load_dotenv


def test_load_dotenv_without_overwriting_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A=from_file\nB='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("A", "existing")

    load_dotenv(env_file)

    assert os.environ["A"] == "existing"
    assert os.environ["B"] == "quoted"
