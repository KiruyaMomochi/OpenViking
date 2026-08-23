# OpenViking user/default trajectory link cleanup investigation

Task: t_62b466e3
Date: 2026-07-03
Scope: read-only investigation; no production data was modified.

## Summary

The noisy log line comes from `AgentExperienceContextProvider._load_source_trajectories()` reading `derived_from` links stored inside experience memory files. The bad source list is not generated at read time by search/vector metadata; it is persisted in the `MEMORY_FIELDS.links` JSON of `viking://user/kyaru/memories/experiences/*.md`.

After the scope migration, the files themselves live under the new scope (`viking://user/kyaru/...`, on disk `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/...`), but their embedded StoredLink endpoints still say `viking://user/default/...`. Reads through the new URI succeed, while reads through the old URI fail with NOT_FOUND. That exactly explains the observed `Failed to read source trajectory viking://user/default/memories/trajectories/<name>.md: File not found` during session commit memory extraction.

## Evidence

### Code path

1. `openviking/session/compressor_v2.py:574-625` runs Phase 2 experience consolidation once per newly written trajectory.
2. `AgentExperienceContextProvider.prefetch()` in `openviking/session/memory/agent_experience_context_provider.py:142-150` searches candidate experiences under `_render_experience_dir(ctx)`.
3. `_render_experience_dir(ctx)` in `agent_experience_context_provider.py:74-88` renders the experiences schema directory with `ctx.user.user_id`; for the current OV config that should be `viking://user/kyaru/memories/experiences`.
4. For top candidates, `prefetch()` calls `_load_source_trajectories(exp_uri, mf.links, viking_fs, ctx)` at `agent_experience_context_provider.py:210-213`.
5. `_load_source_trajectories()` builds its source list directly from `mf.links`: `link_type == "derived_from"`, `to_uri` truthy (`agent_experience_context_provider.py:98-104`), then calls `viking_fs.read_file(uri, ctx=ctx)` (`agent_experience_context_provider.py:106-115`).
6. `mf.links` is parsed from the file's `<!-- MEMORY_FIELDS ... -->` JSON by `MemoryFileUtils.read()` (`openviking/session/memory/utils/memory_file_utils.py:87-92`).

So the immediate source list is: experience file `MEMORY_FIELDS.links[*].to_uri` where `link_type == derived_from`.

### Where those links are written

The links are system-managed, not LLM output:

- `SessionCompressorV2._append_trajectory_metadata()` creates `StoredLink(from_uri=exp_uri, to_uri=traj_uri, link_type="derived_from")` and merges it into `mf.links` (`openviking/session/compressor_v2.py:1038-1068`).
- It also calls `write_stored_links()` to write backlinks on trajectory files (`openviking/session/compressor_v2.py:1074-1075`).
- `write_stored_links()` writes forward links/backlinks to endpoint files (`openviking/session/memory/memory_updater.py:73-111`).
- Supersedes inheritance copies old `derived_from` `to_uri`s from a replaced experience into the new experience (`openviking/session/compressor_v2.py:976-984`).

If an experience file already had old-scope StoredLinks before/during migration, they remain in the file until explicitly rewritten.

### Disk/API verification

Commands run read-only:

- `ov ls viking://user/kyaru/memories/experiences -o json` returned many experience files.
- `ov ls viking://user/default/memories/experiences -o json` returned an empty list.
- `ov ls viking://user/kyaru/memories/trajectories -o json` returned trajectory files.
- `ov read viking://user/kyaru/memories/trajectories/上游代碼歸屬確認_20260626035443.md -o json` succeeded.
- `ov read viking://user/default/memories/trajectories/上游代碼歸屬確認_20260626035443.md -o json` failed with `NOT_FOUND: File not found`.

Representative on-disk path:

- New scope trajectory exists: `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/trajectories/上游代碼歸屬確認_20260626035443.md`
- Old scope trajectory directory has no matching files: `/home/kyaru/.local/share/openviking/data/viking/default/user/default/memories/trajectories`

Representative stale link in a new-scope experience file:

`/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/experiences/upstream代碼歸屬確認.md` contains:

- `from_uri`: `viking://user/default/memories/experiences/upstream代碼歸屬確認.md`
- `to_uri`: `viking://user/default/memories/trajectories/上游代碼歸屬確認_20260626035443.md`
- `link_type`: `derived_from`

But the file itself is now located under `user/kyaru`, and the target trajectory exists under `user/kyaru`.

### Scope of stale references found

Using `search_files` on the OV data subtree only (not recursive over `/home/kyaru`):

- In `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/experiences`, `viking://user/default/memories/trajectories/` appears 235 times across many experience files.
- In the same experience subtree, `viking://user/default/memories/experiences/` also appears 235 times, mostly stale `from_uri` values in the same StoredLinks.
- In `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/trajectories`, stale backlinks likewise appear: 238 matches for old-scope trajectory URIs and 238 matches for old-scope experience URIs.
- `viking://user/default/memories/experiences` via OV API is empty, confirming these links are not pointing to live old-scope files.

## Upstream check

No matching public GitHub/web result was found for this exact `user/default` -> `user/kyaru` stale StoredLink migration issue or `_load_source_trajectories` failure. Repository history does show related upstream work:

- `28fb9d9b feat(memory): replace source_trajectories with StoredLink for trajectory-experience linking (#2222)` introduced/stabilized StoredLink-based trajectory-experience linking.
- `e718e507 Fix noisy source trajectory load errors` added filtering for trajectory URIs and changed the log from `tracer.error` to `logger.warning`, but it does not remap stale user scopes. The current file in this checkout has the noisy `tracer.error` form again, likely due to later branch history/rebase.

## Root cause

This is stale persisted data from the scope migration, not a current search fallback bug.

The current extraction code reads experience candidates from `user/kyaru` correctly. The bad `user/default` strings are inside persisted StoredLinks (`links` on experiences and `backlinks` on trajectories). VikingFS uses pure URI/path mapping for normal user URIs (`viking://...` -> `/local/{account}/...`) and only has special read fallbacks for legacy sessions/agents, not for arbitrary old user namespaces. Therefore `viking://user/default/...` correctly resolves to the now-empty old user namespace and fails.

## Recommended remediation (do not run without production approval)

Safest data fix: perform a targeted StoredLink rewrite under the OV data root, changing only StoredLink endpoint fields from the old user memory namespace to the new one:

- `viking://user/default/memories/trajectories/` -> `viking://user/kyaru/memories/trajectories/`
- `viking://user/default/memories/experiences/` -> `viking://user/kyaru/memories/experiences/`

Apply this to both:

- `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/experiences/**/*.md`
- `/home/kyaru/.local/share/openviking/data/viking/default/user/kyaru/memories/trajectories/**/*.md`

After rewriting:

1. Verify every rewritten `to_uri` / `from_uri` can be read or is intentionally absent.
2. Run targeted reindex for the affected subtree, e.g. `ov reindex viking://user/kyaru/memories --mode semantic_and_vectors --wait true` (or narrower `experiences` + `trajectories` if supported/desired).
3. Commit a new session and confirm `journalctl --user -u openviking` no longer shows missing old-scope source trajectory reads.

Operationally safer implementation shape:

- first run a dry-run script that parses only `MEMORY_FIELDS` JSON and prints planned replacements + files touched;
- create a backup archive of the affected `experiences` and `trajectories` directories;
- stop or pause the OpenViking service/extraction workers during rewrite to avoid racing with memory updates;
- rewrite via OV API if a batch edit endpoint exists, otherwise carefully patch the markdown files on disk;
- reindex after service resumes.

## Optional code hardening

Data cleanup should be the primary fix. If you want extra protection against future migrations, add a repair/normalization layer rather than silently broad fallback:

- In `_load_source_trajectories`, when a derived_from URI 404s and matches `viking://user/default/memories/trajectories/`, try the same basename under the current `ctx.user.user_id` only as a read fallback, and log a warning that persisted links need migration.
- Alternatively, add a dedicated migration command that rewrites StoredLinks under a subtree. This is cleaner than embedding user/default fallback into runtime reads forever.

I would not add a general VikingFS fallback from `user/default` to current user; that could mask genuinely separate user namespaces and violate isolation semantics.

## Notes / blockers

- I did not modify production data, per task instructions.
- `uv run python` in this checkout attempted to build OpenViking editable and failed because the environment lacks cargo/make for the native extensions. I used existing CLI, file reads, git, and `search_files` instead.
- A Node one-off script would have been convenient for aggregate counts, but the terminal approval guard blocked heredoc script execution; all production inspection above was done with approved read/search tools and OV CLI reads/lists.
