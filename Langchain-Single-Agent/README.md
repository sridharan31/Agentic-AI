# Langchain Agent

A minimal LangChain-based agent project built for experimentation and learning. This repository demonstrates how to set up a Python environment, install dependencies, and run a simple agent-driven application using environment-based API keys.

## Features

- Simple LangChain agent scaffold
- Environment variable support via `.env`
- Example app entrypoints: `app.py`, `main.py`
- Notebook research work in `research/agent_demo.ipynb`

## Technology

- Python 3.11
- LangChain
- Ollama (local LLM)
- dotenv for environment configuration

## Architecture

This repository is organized as a small Python agent project:

- `app.py` - Primary application script for launching the agent logic.
- `main.py` - Secondary entrypoint, typically used for testing or demo execution.
- `ollama_agent.py` - Shared Ollama tool-calling workflow used by the app, CLI, and notebook.
- `requirements.txt` - Python dependencies required to run the project.
- `.env` - Local optional API keys for web search and weather.
- `research/agent_demo.ipynb` - Jupyter notebook for experimentation and development notes.

The application follows a simple architecture:

1. Load configuration from `.env`.
2. Initialize the agent and language model client.
3. Execute the agent workflow through `app.py` or `main.py`.

## Getting Started

### Prerequisites

- Python 3.11
- `conda` or any Python virtual environment manager

### Installation

For Windows PowerShell, use the project's Python 3.11 environment. The pinned
LangChain 0.1 dependencies require NumPy 1.x, which does not have a compatible
wheel for Python 3.13 and may fail while trying to compile from source.

If `.venv311` already exists, install and run Streamlit through that interpreter:

```powershell
& ".\.venv311\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv311\Scripts\python.exe" -m streamlit run app.py
```

For notebook kernel selection and troubleshooting, see
[Notebook Kernel Troubleshooting](./NOTEBOOK_KERNEL_TROUBLESHOOTING.md).

To create the environment from scratch:

```powershell
py -3.11 -m venv .venv311
& ".\.venv311\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv311\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv311\Scripts\python.exe" -m streamlit run app.py
```

On macOS/Linux, create and activate a Python 3.11 environment before installing
the requirements:

```bash
conda create -n langagent python=3.11 -y
conda activate langagent
pip install -r requirements.txt
```

### Configuration

The LLM runs locally with Ollama, so it does not require a hosted-model API key.
Install Ollama, then ensure the model used by the project is available:

```bash
ollama pull llama3.2
```

Create a `.env` file at the repository root for optional web-search and weather tools:

```dotenv
TAVILY_API_KEY="your-tavily-api-key"
WEATHERSTACK_API_KEY="your-weatherstack-api-key"
```

> Do not commit your `.env` file or secret keys to source control.

Without a Tavily key, the agent runs without web search. Ollama must be running locally
at its default endpoint (`http://localhost:11434`).
The agent uses Ollama's native tool-call API, avoiding the text-format parser required by
the older LangChain ReAct prompt. Set `OLLAMA_BASE_URL` only if Ollama is running at a
non-default endpoint. The app exposes weather tools for weather questions and web-search
tools for explicit or recent-information searches; unrelated questions are sent as plain
chat so the local model does not make spurious tool calls.

### Run the Project

Use one of the Python entrypoints:

```bash
python app.py
```

or

```bash
python main.py
```

### Notebook Exploration

Open the research notebook for additional agent demos and experiments:

```bash
jupyter notebook research/agent_demo.ipynb
```

## Project Structure

```text
.
├── .env
├── README.md
├── app.py
├── main.py
├── requirements.txt
└── research/
    └── agent_demo.ipynb
```

## Contribution

Contributions are welcome. If you'd like to improve this repo, open an issue or submit a pull request with enhancements, bug fixes, or documentation improvements.

## License

This project is provided under an open source-friendly license. Add a license file if you want to publish this repository publicly.
