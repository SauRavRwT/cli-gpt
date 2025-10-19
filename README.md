## CLI-GPT

A small command-line chat client for multiple LLM providers (Gemini and Groq) using the OpenAI-style client API.

This repository contains `app.py`, a simple interactive CLI that sends prompts to a configured model, shows rich-formatted responses, and saves conversations.

## Features

- Interactive, readline-style prompt with persistent history
- Two built-in models: `gemini` and `llama` (configured to use Gemini and Groq providers)
- Save and load conversation transcripts (JSON) under `./saved/conversations`
- Simple session commands: `/save`, `/clear`, `/model`, `/switch`, `/help`, and `exit`
- Rich output formatting (Markdown, panels)

## Requirements

- Python 3.10+
- A virtual environment is recommended

Required Python packages (example):

- openai (or a compatible client used in `app.py`)
- python-dotenv
- rich
- prompt-toolkit

You can create a minimal `requirements.txt` with:

```
openai
python-dotenv
rich
prompt-toolkit
```

Install with pip:

```bash
python -m venv venv
source venv/bin/activate   # on Linux
venv\Scripts\activate   # on Windows (cmd)
pip install -r requirements.txt
```

## Configuration

This app expects two API keys to be set in a `.env` file in the project root:

- `GEMINI_API_KEY` — API key for the Gemini provider
- `GROQ_API_KEY` — API key for the Groq provider

Create a `.env` file like this:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

If either key is missing the app will raise an error at startup.

## Run

Start the CLI with:

```bash
python app.py
```

Optional: pick a model at startup (defaults to `gemini`):

```bash
python app.py --model llama
```

## Interactive commands

While running, use the following commands:

- `/save` — Save the current conversation to `./saved/conversations/<timestamp>.json`
- `/clear` — Clear the current in-memory conversation and start fresh
- `/model` — Show the currently selected model
- `/switch` — Switch to the next model in the configured list (cycles)
- `/help` — Display help
- `exit` or `quit` — Save conversation and exit

Prompts you type are sent to the configured model; responses are displayed using rich formatting and Markdown rendering.

## Conversation storage

Conversations are stored as JSON files in `./saved/conversations/` with the filename format `YYYYMMDD_HHMMSS.json` and contain:

- `model` — the model name used
- `timestamp` — ISO timestamp of the conversation start
- `messages` — array of role/content messages (system, user, assistant)

The CLI also keeps a simple command history file at `./saved/command_history` used by prompt-toolkit to persist input history across sessions.

## Troubleshooting

- If you see errors about missing environment variables, ensure `.env` contains `GEMINI_API_KEY` and `GROQ_API_KEY` and that you restarted the shell or reactivated the venv.
- If the model raises `model_decommissioned` or `model_not_found`, the CLI will surface a helpful error message. Update `MODELS` in `app.py` to change model names or add new providers.
- If responses are not returned, check your API keys and network connectivity.

## Notes & Next steps

- Consider adding a `requirements.txt` or `pyproject.toml` to pin dependencies.
- Add an example `.env.example` file (without keys) for convenience.
- Add tests for the conversation saving logic and a small integration test that mocks the API clients.
