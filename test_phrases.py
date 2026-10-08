"""
Run:  python test_phrases.py
Checks that every phrase shown in the voice manual really produces the command
the manual promises (offline: phrasebook + regex), and that the LLM path
validates and falls back correctly (mocked, no API key needed).
"""
import os, sys, types
os.environ.pop("ANTHROPIC_API_KEY", None)
import server as S

SUB = {"<folder>": "projects", "<file>": "notes dot txt", "<text>": "hello world"}
EXP = {"<folder>": "projects", "<file>": "notes.txt",     "<text>": "hello world"}


def fill(s, table):
    for k, v in table.items():
        s = s.replace(k, v)
    return s


fails, total = [], 0
for entry in S.PHRASEBOOK:
    for g in entry["groups"]:
        for p in g["phrases"]:
            total += 1
            say, want = fill(p, SUB), fill(g["expect"], EXP)
            got = S._phrasebook(say) or S._apply_regex(say)
            if got != want:
                fails.append((entry["id"], say, want, got))

# natural variations people actually say / STT produces
EXTRA = {
    "Where am I?": "pwd", "WHERE AM I": "pwd", "where's this": "pwd",
    "could you please tell me where I am?": "pwd",
    "What's in here?": "ls", "list the files for me please": "ls",
    "ls -la": "ls -la", "list files in docs": "ls docs",
    "what's in notes dot txt": "cat notes.txt",
    "read tilde slash notes dot txt": "cat ~/notes.txt",
    "open notes.txt": "cat notes.txt", "open the projects folder": "cd projects",
    "go to docs slash src": "cd docs/src", "go to my documents": "cd documents",
    "go to project files": "cd 'project files'",
    "cat my underscore file dot txt": "cat my_file.txt",
    "go to double dot": "cd ..", "go to the home directory": "cd ~",
    "say what's up": "echo what's up", "say thank you": "echo thank you",
    "clear the screen please": "cls", "go home please": "cd ~",
}
for say, want in EXTRA.items():
    total += 1
    got = S._normalize_voice(say)[0]
    if got != want:
        fails.append(("extra", say, want, got))

print(f"phrases checked : {total}")
print(f"exact phrases   : {len(S._LIT):,}")
print(f"key conflicts   : {len(S._LIT_CONFLICTS)}")
for c in S._LIT_CONFLICTS[:10]:
    print("   CONFLICT", c)
print(f"failures        : {len(fails)}")
for f in fails:
    print("   FAIL", f)

# ── mocked LLM: parsing, validation, fallback ────────────────────────────────
class _Blk:  # minimal stand-in for an SDK text block
    def __init__(s, t): s.type, s.text = "text", t


def fake(reply, seen=None):
    def create(**kw):
        if seen is not None: seen.append(kw)
        if isinstance(reply, Exception): raise reply
        return types.SimpleNamespace(content=[_Blk(reply)])
    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


os.environ["ANTHROPIC_API_KEY"] = "test"
llm_fail = []
cases = [
    ('{"command":"cd Documents","confidence":"high"}',            ("cd Documents", "llm")),
    ('```json\n{"command":"cat README.md","confidence":"high"}\n```', ("cat README.md", "llm")),
    ('Sure! {"command":"pwd","confidence":"medium"}',             ("pwd", "llm")),
    ('{"command":"rm -rf /","confidence":"high"}',                ("blah", "llm_invalid")),
    ('{"command":"pwd now","confidence":"high"}',                 ("blah", "llm_invalid")),
    ('{"command":"cat","confidence":"high"}',                     ("blah", "llm_invalid")),
    ('{"command":"ls","confidence":"low"}',                       ("blah", "llm_no_match")),
    ('{"command":"","confidence":"low"}',                         ("blah", "llm_no_match")),
    ('not json at all',                                           ("blah", "llm_no_match")),
    (RuntimeError("network down"),                                ("blah", "llm_error")),
]
for reply, want in cases:
    S._LLM = fake(reply)
    got = S._llm_parse("blah")
    if got != want:
        llm_fail.append((reply, want, got))

seen = []
S._LLM = fake('{"command":"cd Documents","confidence":"high"}', seen)
S._normalize_voice("pop into the docs")
sent = seen[0]
assert sent["temperature"] == 0 and "<utterance>pop into the docs</utterance>" in sent["messages"][0]["content"]
assert "<entries>" in sent["messages"][0]["content"] and "cd <folder>" in sent["system"]

# phrasebook hits must not spend an API call; free-form must; LLM failure → regex
seen.clear(); S._normalize_voice("where am i")
assert not seen, "documented phrase should not call the API"
S._LLM = fake(RuntimeError("down"))
assert S._normalize_voice("enter projects")[0::2] == ("cd projects", "regex")

print(f"LLM mock cases  : {len(cases)+3} run, {len(llm_fail)} failed")
for f in llm_fail:
    print("   FAIL", f)
sys.exit(1 if (fails or llm_fail or S._LIT_CONFLICTS) else 0)
