# KRA$H

> **A Unix-inspired terminal that understands typed and natural-language voice commands.**

KRA$H combines a browser terminal, speech-to-text, NLP, and a Python execution engine into **one shared command pipeline**.

Type:

```text
ls
```

or say:

```text
"show me what's in this folder"
```

Both resolve to:

```text
ls
```

### Features

* Typed commands
* Browser-based voice input
*  Natural-language command parsing
* Anthropic NLP fallback
* Python + Flask backend
*  Restricted command allowlist
* Built-in voice/command reference

### Supported Commands

`ls` · `cd` · `pwd` · `echo` · `cat` · `cls`

### Architecture

```text
Keyboard / Voice
       ↓
   index.html
       ↓
    server.py
       ↓
 test_phrases.py
       ↓
Canonical Command
       ↓
Validation → Execution
```

**Voice is an input modality, not a separate shell.**

### Project Structure

```text
KRA$H/
├── index.html
├── server.py
├── test_phrases.py
├── voice-manual.html
└── requirements.txt
```

### Setup

```bash
pip install -r requirements.txt
python server.py
```

Set your Anthropic API key before running:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your-api-key"
```

### Security

KRA$H uses a small command allowlist and validates parsed commands before execution. Natural-language input cannot directly become arbitrary shell code.

---

> **Talk to the shell. Make it understand.**
