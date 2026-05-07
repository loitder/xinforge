# Xinforge

A minimal Rust CLI coding agent backed by the [DeepSeek API](https://api.deepseek.com). It runs a conversational tool-calling loop — the model can read/write/edit files, run shell commands, spawn sub-agents, and track tasks with a todo list.

## Quick start

```bash
# Clone and build
git clone <repo-url> && cd xinforge
cargo build --release

# Set credentials (or pass via env)
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY="sk-your-key-here"
export MODEL_ID="deepseek-v4-flash"

cargo run
```

Type your request at the `>>` prompt. Type `q` or `exit` to quit.

## How it works

1. You type a natural-language request.
2. The agent sends it to DeepSeek with a set of built-in tools.
3. If the model returns tool calls the agent executes them (bash, file I/O, todo, sub-agent spawn, …) and feeds the results back.
4. The loop continues until the model produces a final text answer.

## Project layout

```
src/
├── main.rs      # CLI entry point, prompt loop
├── agent.rs     # Agent orchestrator — tool-call loop,
│                #   reasoning tracking, compaction
├── llm.rs       # DeepSeek API client (reqwest + serde_json)
├── tools.rs     # Tool definitions & implementations
├── todo.rs      # Structured todo list tracker
└── skill.rs     # Skill loader (YAML frontmatter SKILL.md files)
```

## Available tools

| Tool | Description |
|------|-------------|
| `bash` | Run a shell command (dangerous patterns blocked) |
| `read_file` | Read file contents |
| `write_file` | Write content to a file |
| `edit_file` | Replace exact text in a file |
| `todo` | Update structured task list |
| `load_skill` | Load skill knowledge from `skills/<name>/SKILL.md` |
| `task` | Spawn a sub-agent for parallel work |
| `compact` | Trigger conversation compression |

## Skills

Drop `SKILL.md` files under `skills/<name>/`. The frontmatter block declares the skill name and description:

```markdown
---
name: my-skill
description: Documentation for my project
---

Full skill body here…
```

Skills are injected into the system prompt (descriptions) and loaded on demand via the `load_skill` tool.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL1` | `https://api.deepseek.com/v1` | API base URL |
| `OPENAI_API_KEY1` | — | API key |
| `MODEL_ID` | `deepseek-v4-flash` | Model to use |

## Compaction

When the conversation exceeds ~50k tokens the agent automatically summarizes earlier turns into a transcript file under `.transcripts/`. You can also trigger it manually with the `compact` tool.
