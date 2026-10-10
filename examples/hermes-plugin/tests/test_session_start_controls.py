"""Session-start switches retain profile-bound config and request suppression."""

import pytest
import yaml

from .test_gateway_recall import profile_scope


@pytest.mark.parametrize("profile,memories", [(False, False), (True, False), (False, True), (True, True)])
def test_session_start_controls(external_provider, monkeypatch, profile, memories):
    home, provider, _, _ = external_provider("controls")
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": {"openviking": {
        "session_start_profile": profile, "session_start_memories": memories,
    }}}))
    requests = []

    class Client:
        def get(self, path, **kwargs):
            requests.append(path)
            if path.endswith("read"):
                return {"result": {"content": "Profile fact"}}
            return {"result": [{"name": f"{n}.md", "abstract": "fact"} for n in range(5)]}

    provider._client = Client()
    monkeypatch.setattr(provider, "_user_space", lambda *args, **kwargs: "default")
    with profile_scope(home):
        result = provider._session_start_memory_context("session")
        assert ("Profile fact" in result) == profile
        assert ("<available-memories>" in result) == memories
        assert requests.count("/api/v1/content/read") == int(profile)
        assert requests.count("/api/v1/fs/ls") == 2 * int(memories)


def test_env_overrides_yaml(external_provider):
    home, provider, _, _ = external_provider("override")
    (home / "config.yaml").write_text(
        "memory: {openviking: {session_start_profile: false, session_start_memories: true, available_memories_limit: 7}}\n")
    (home / ".env").write_text(
        "OPENVIKING_SESSION_START_PROFILE=true\n"
        "OPENVIKING_SESSION_START_MEMORIES=false\n"
        "OPENVIKING_AVAILABLE_MEMORIES_LIMIT=2\n")
    with profile_scope(home):
        assert provider._session_start_config() == {"profile": True, "memories": False, "available_memories_limit": 2}


def test_listing_entry_limit_and_tail(external_provider):
    _, provider, _, _ = external_provider("budget")
    entries = [{"name": f"{n}.md", "abstract": "long fact" * 4} for n in range(10)]
    lines, _ = provider._format_memory_listing("viking://user/default/preferences", entries, 10000, max_entries=2)
    shown = sum(line.startswith("    - ") for line in lines)
    assert shown == 2
    assert any("+8 more" in line for line in lines)
