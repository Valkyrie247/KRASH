# KRA$H

> **A Unix-inspired terminal that understands both typed commands and natural language voice commands.**

KRA$H is a browser-based Unix-inspired terminal interface that combines **keyboard input, browser speech-to-text, natural-language command parsing, and a Python execution engine** into a single command pipeline.

Instead of maintaining separate typed and voice shells, KRA$H treats both modalities as different input sources for the **same shell**.

A user can type:

```text
ls
```

or say:

```text
"show me what's in this folder"
```

Both are ultimately resolved to the same canonical command:

```text
ls
```

and passed through the same execution engine.

---

## ✦ Features

* ⌨️ **Typed terminal commands**
* 🎙️ **Browser-based voice commands**
* 🧠 **Natural-language command interpretation**
* 🤖 **Anthropic-powered NLP fallback**
* 🔀 **Single shared command pipeline**
* 🐍 **Python + Flask backend**
* 💻 **Unix-style command execution**
* 🔒 **Restricted command allowlist**
* 📁 Persistent single-session working directory
* 🖥️ Terminal-style UI
* 📖 Built-in phrase reference / man page

---

## Architecture

KRA$H deliberately avoids creating separate shells for keyboard and voice input.

Both inputs enter the same pipeline:

```text
                     ┌──────────────────┐
                     │   Keyboard Input │
                     └────────┬─────────┘
                              │
                              │
                              ▼
                     ┌──────────────────┐
                     │    index.html    │
                     │  Terminal UI     │
                     └────────┬─────────┘
                              │
                     ┌────────┴─────────┐
                     │                  │
                     ▼                  ▼
                Typed text        Browser STT
                     │                  │
                     └────────┬─────────┘
                              ▼
                       POST /execute
                              │
                              ▼
                     ┌──────────────────┐
                     │    server.py     │
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │ test_phrase.py   │
                     │                  │
                     │ NLP / Intent     │
                     │ Parser           │
                     └────────┬─────────┘
                              │
                              ▼
                     Canonical command
                              │
                              ▼
                     Command validation
                              │
                     ┌────────┴─────────┐
                     ▼                  ▼
                  Built-ins        subprocess
                     │                  │
                     └────────┬─────────┘
                              ▼
                         Terminal output
```

The important design principle is:

> **Voice is an input modality, not a separate shell.**

---

## Supported Commands

KRA$H currently supports six commands:

| Command | Purpose                           |
| ------- | --------------------------------- |
| `ls`    | List directory contents           |
| `cd`    | Change working directory          |
| `pwd`   | Display current working directory |
| `echo`  | Print text                        |
| `cat`   | Display file contents             |
| `cls`   | Clear terminal output             |

### Examples

The same command can be expressed in different ways.

#### `ls`

```text
ls
list files
show the files
show me what's in this folder
what files are here
display the contents of this directory
```

#### `pwd`

```text
pwd
where am I
what directory am I in
show my current path
tell me my current working directory
```

#### `cd`

```text
cd projects
go to projects
move to the projects folder
change to projects
```

#### `cat`

```text
cat notes.txt
read notes.txt
show the contents of notes.txt
display what's inside notes.txt
```

---

## Natural Language Processing

KRA$H uses a two-stage parsing strategy.

### 1. Deterministic phrase matching

Common phrases are resolved locally without an API call.

For example:

```text
"where am I?"
```

is directly mapped to:

```json
{
  "command": "pwd",
  "arguments": []
}
```

This makes common commands fast and predictable.

### 2. Anthropic NLP fallback

If deterministic matching cannot confidently understand an input, the request can be passed to Anthropic for intent interpretation.

For example:

```text
"Could you tell me what's currently inside this folder?"
```

can become:

```json
{
  "command": "ls",
  "arguments": []
}
```

The model is constrained to the KRA$H command vocabulary.

It cannot return arbitrary shell commands.

---

## Why an NLP Layer?

Traditional terminals expect users to know exact syntax:

```text
ls
cd Downloads
pwd
cat notes.txt
```

KRA$H explores a different interaction model:

```text
"show me the files"
"take me to Downloads"
"where am I?"
"read notes.txt"
```

The NLP layer acts as an **intent-to-command translator**, while the execution engine remains responsible for actually running the command.

This separation keeps language understanding independent from command execution.

---

## Security Model

KRA$H does **not** expose an unrestricted shell.

The execution engine only accepts:

```text
ls
cd
pwd
echo
cat
cls
```

The NLP layer produces a structured command rather than executable shell code.

Before execution, the command is validated against the allowlist.

For external commands, KRA$H uses:

```python
subprocess.run(
    [command, *arguments],
    ...
)
```

rather than:

```python
subprocess.run(command, shell=True)
```

This prevents natural-language input from directly becoming arbitrary shell execution.

Shell operators such as:

```text
;
&&
||
|
>
<
$()
```

are rejected by the parser.

---

## `cd` as a Built-in

`cd` is implemented inside the Python application rather than through `subprocess`.

This is important because a child process cannot change the working directory of its parent Python process.

KRA$H therefore maintains:

```python
_cwd
```

as its session working directory.

Commands such as `ls` and `cat` are then executed relative to that directory.

---

## Frontend

The terminal interface uses:

* **JetBrains Mono**
* Deep navy terminal background
* Warm amber prompts
* Sky-blue `echo` output
* Soft gray terminal output
* Rose voice/listening state

The interface contains:

```text
┌──────────────────────────────────────────────┐
│ KRA$H                         ● SESSION ACTIVE│
├──────────────────────────────────────────────┤
│                                              │
│ Kra$h:/home/user$ ls                         │
│ file.txt                                     │
│ projects/                                    │
│                                              │
│ 🎙 VOICE Kra$h:/home/user$ show the files    │
│ file.txt                                     │
│ projects/                                    │
│                                              │
├──────────────────────────────────────────────┤
│ Kra$h:/home/user$                    🎙      │
└──────────────────────────────────────────────┘
```

Voice-originated commands receive a visual microphone badge so that users can distinguish them from typed commands.

---

## Project Structure

```text
KRA$H/
│
├── index.html
├── server.py
├── test_phrase.py
├── phrases
└── requirements.txt
```

### `index.html`

The only frontend.

Responsible for:

* Terminal UI
* Keyboard input
* Browser speech recognition
* Sending requests to Flask
* Rendering command/output history
* Displaying voice-originated commands

### `server.py`

The Flask backend and execution engine.

Responsible for:

* `/execute`
* `/cwd`
* Command routing
* Built-in commands
* Working-directory management
* `subprocess` execution
* Response generation

### `test_phrase.py`

The natural-language parser.

Responsible for:

* Canonical command recognition
* Phrase matching
* Anthropic NLP fallback
* Intent extraction
* Command/argument validation

### `phrases`

A man-page-style reference documenting:

* Supported commands
* Typed syntax
* Natural-language alternatives
* Example phrases
* NLP pipeline

---

## API

### `POST /execute`

Request:

```json
{
  "command": "show me the files",
  "source": "voice"
}
```

Response:

```json
{
  "out": "...",
  "output": "...",
  "cwd": "/current/path",
  "source": "voice",
  "input": "show me the files",
  "parsed_command": "ls",
  "arguments": []
}
```

Both keyboard and voice requests use this same endpoint.

---

### `GET /cwd`

Returns the current KRA$H working directory.

```json
{
  "cwd": "/current/path"
}
```

---

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd KRA$H
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Anthropic API key.

### Windows PowerShell

```powershell
$env:ANTHROPIC_API_KEY="your-api-key"
```

### Linux/macOS

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Start the server:

```bash
python server.py
```

Then open the terminal frontend through the Flask server.

---

## Example Interaction

### Typed

```text
Kra$h:/home/user$ pwd
/home/user

Kra$h:/home/user$ ls
notes.txt
projects

Kra$h:/home/user$ cd projects

Kra$h:/home/user/projects$ ls
krash
ciphantom
```

### Voice

```text
🎙 "Where am I?"

→ pwd

/home/user
```

```text
🎙 "Show me what's inside this folder"

→ ls

notes.txt
projects
```

```text
🎙 "Go to projects"

→ cd projects
```

The voice commands ultimately execute through the **same command pipeline as their typed equivalents**.

---

## Design Philosophy

KRA$H is built around three principles:

### 1. One shell

There is no "voice shell" and "keyboard shell."

There is one shell with multiple input modalities.

### 2. Language ≠ execution

Natural-language processing determines **what the user means**.

The execution engine determines **what is allowed to run**.

### 3. Constrain before execution

The NLP model operates inside a deliberately tiny command vocabulary.

The model cannot decide to execute arbitrary operating-system commands.

---

## Future Possibilities

Potential extensions include:

* Command history
* Tab completion
* More sophisticated argument parsing
* Persistent sessions
* Multi-user sessions
* Better voice feedback
* Offline speech recognition
* Additional Unix commands
* Pipeline support
* Command aliases
* CTF-style hidden functionality
* More advanced intent classification

---

## Tech Stack

**Frontend**

* HTML
* CSS
* JavaScript
* Browser Speech Recognition API
* JetBrains Mono

**Backend**

* Python
* Flask
* `subprocess`

**NLP**

* Deterministic phrase matching
* Anthropic API

---

## Project Status

🚧 **Active development**

KRA$H is currently a constrained Unix-inspired terminal prototype exploring how **natural-language and voice interaction can coexist with a traditional command-line execution model**.

---

> **KRA$H**
>
> *Talk to the shell. Make it understand.*

