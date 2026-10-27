# LangChain Multi-Agent Research System

A powerful multi-agent research system built with LangChain that autonomously researches topics, gathers information, writes comprehensive reports, and evaluates their quality using AI-powered agents.

<p align="center">
  <strong>🔬 Research Automation • 🤖 Multi-Agent Orchestration • 📝 Intelligent Report Generation</strong>
</p>

---

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents for searching, reading, writing, and critiquing
- **Automated Web Research**: Intelligent web search with Tavily API
- **Smart Content Extraction**: Advanced web scraping with multiple fallback strategies
- **AI-Powered Report Generation**: Automatically generates structured research reports
- **Quality Evaluation**: Built-in critic agent for report validation and scoring
- **Interactive UI**: Streamlit-based user interface for easy interaction
- **Pipeline Orchestration**: Seamless coordination of multiple agents

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           Streamlit UI (app.py)                     │
│      Multi-Agent Research Assistant Interface       │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│      Research Pipeline (pipeline.py)                │
│        Orchestrates multi-agent workflow            │
└──────────────────┬──────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼─────┐   ┌───▼────┐
│Search │    │   Reader  │   │ Writer │
│Agent  │    │   Agent   │   │ Chain  │
└───┬───┘    └────┬─────┘   └───┬────┘
    │             │             │
    │  ┌──────────▼─────────┐   │
    └─▶│  Tools Layer       │◀──┘
       │                    │
       │ • web_search      │
       │ • scrape_url      │
       │                    │
       └────────┬───────────┘
                │
            ┌───▼────────┐
            │ Critic     │
            │ Chain      │
            └────────────┘
```

### Agent Responsibilities

- **Search Agent**: Discovers relevant information across the web using Tavily
- **Reader Agent**: Extracts clean, readable content from URLs
- **Writer Chain**: Composes structured, professional research reports
- **Critic Chain**: Evaluates reports and provides improvement suggestions

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **LangChain** | Multi-agent orchestration and chain management |
| **OpenAI GPT-4o-mini** | Language model for agents and chains |
| **Streamlit** | Interactive web UI |
| **Tavily API** | Web search and information retrieval |
| **BeautifulSoup4** | HTML parsing and content extraction |
| **Trafilatura** | Web content extraction |
| **Readability-lxml** | Article content extraction |
| **Python-dotenv** | Environment configuration management |
| **Rich** | Terminal output formatting |

---

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API Key
- Tavily API Key

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LangChain-Multi-Agent-Research-System.git
cd LangChain-Multi-Agent-Research-System
```

### 2. Create Environment (Conda)

```bash
conda create -n langagent python=3.11 -y
conda activate langagent
```

Or with venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys from:
- [OpenAI API](https://platform.openai.com/api-keys)
- [Tavily API](https://tavily.com)

---

## 💡 Usage

### Run with Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Run as a Script

```bash
python main.py
```

Edit the `topic` variable in `main.py` to research different topics.

---

## 📁 Project Structure

```
.
├── app.py                      # Streamlit web interface
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # License file
├── demo.excalidraw            # Architecture diagram
│
└── src/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   └── agents.py           # Search, Reader, Writer, Critic agents
    ├── tools/
    │   ├── __init__.py
    │   └── tools.py            # web_search, scrape_url tools
    └── pipelines/
        ├── __init__.py
        └── pipeline.py         # Main research orchestration
```

---

## 🔄 Workflow

1. **User Input**: Enter a research topic via UI or script
2. **Search Phase**: Search agent queries the web using Tavily
3. **Reading Phase**: Reader agent extracts content from relevant URLs
4. **Writing Phase**: Writer chain synthesizes findings into a structured report
5. **Review Phase**: Critic chain evaluates the report and provides scores
6. **Output**: Display final report with feedback and scores

---

## 🔗 LCEL (LangChain Expression Language)

The Writer and Critic steps in this project are built with **LCEL**. This section explains what LCEL is, how it works, and how this project uses it.

### What is LCEL?

LCEL is a declarative way to build LLM pipelines by **joining components with the pipe operator `|`**, like a Unix shell pipeline:

```python
chain = prompt | llm | parser
```

The output of each step becomes the input of the next. The result, `chain`, is itself a component you can run, stream, batch, or nest inside a bigger chain.

LCEL replaces the legacy chain classes (`LLMChain`, `SequentialChain`, `SimpleSequentialChain`), which are deprecated in LangChain 1.x.

### Why use LCEL?

| Benefit | What you get |
|---------|--------------|
| **One interface everywhere** | Every chain has `invoke`, `stream`, `batch`, and the async forms `ainvoke`, `astream`, `abatch` |
| **Streaming for free** | Tokens stream through the whole chain, so the UI can show text as it is generated |
| **Parallelism** | `batch()` and `RunnableParallel` run work concurrently without threading code |
| **Composability** | A chain can be a step in another chain |
| **Reliability** | Built-in `.with_retry()` and `.with_fallbacks()` |
| **Observability** | Every step is traced automatically in LangSmith |
| **Readable** | The pipeline reads left to right in one line |

### The core idea: the Runnable

Everything in LCEL is a **Runnable**, an object that implements the same standard methods:

| Method | What it does |
|--------|--------------|
| `invoke(input)` | Run once, return the full output |
| `stream(input)` | Yield output chunks as they are produced |
| `batch([inputs])` | Run many inputs concurrently |
| `ainvoke` / `astream` / `abatch` | Async versions for web servers (FastAPI, etc.) |

Prompts, chat models, output parsers, retrievers, tools, and agents are all Runnables. Because they share one interface, any of them can be joined with `|`. Writing `a | b` creates a `RunnableSequence`, which is also a Runnable.

### How this project uses LCEL

In [src/agents/agents.py](src/agents/agents.py):

```python
writer_chain = writer_prompt | llm | StrOutputParser()
critic_chain = critic_prompt | llm | StrOutputParser()
```

Here is what happens inside `writer_chain.invoke({"topic": ..., "research": ...})`:

```
 {"topic": "...", "research": "..."}        ← Python dict
            │
            ▼
 ┌─────────────────────────┐
 │ writer_prompt           │  ChatPromptTemplate: fills {topic} and {research}
 │ (ChatPromptTemplate)    │
 └───────────┬─────────────┘
             │  ChatPromptValue (system + human messages)
             ▼
 ┌─────────────────────────┐
 │ llm                     │  ChatOpenAI("gpt-4o-mini") calls the API
 │ (ChatOpenAI)            │
 └───────────┬─────────────┘
             │  AIMessage(content="# Introduction ...")
             ▼
 ┌─────────────────────────┐
 │ StrOutputParser()       │  pulls .content out of the AIMessage
 └───────────┬─────────────┘
             │
             ▼
     "# Introduction ..."                    ← plain str
```

| Step | Input type | Output type |
|------|-----------|-------------|
| `writer_prompt` | `dict` | `ChatPromptValue` |
| `llm` | `ChatPromptValue` | `AIMessage` |
| `StrOutputParser()` | `AIMessage` | `str` |

Each step's output type must match the next step's input type. Most LCEL errors come from a mismatch here.

The pipeline in [src/pipelines/pipeline.py](src/pipelines/pipeline.py) calls the chains like this:

```python
state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
state["feedback"] = critic_chain.invoke({"report": state["report"]})
```

**Chains vs agents in this project:** the Search and Reader steps use `create_agent` because the model must *decide* which tool to call and when. The Writer and Critic steps always do the same thing (fill a prompt, call the model, return text), so a fixed LCEL chain is simpler, faster, and cheaper.

### LCEL building blocks

All of these are imported from `langchain_core.runnables`.

#### 1. `RunnableSequence`: run steps one after another

You rarely create it directly; `|` builds it for you.

```python
chain = writer_prompt | llm | StrOutputParser()
print(type(chain).__name__)   # RunnableSequence
```

#### 2. `RunnableParallel`: run steps at the same time on the same input

Returns a dict with one key per branch. A plain dict inside a chain is shorthand for `RunnableParallel`.

```python
from langchain_core.runnables import RunnableParallel

# Two critics review the same report concurrently
reviews = RunnableParallel(
    critique=critic_chain,
    summary=summary_prompt | llm | StrOutputParser(),
)
result = reviews.invoke({"report": report})
# {"critique": "Score: 8/10 ...", "summary": "..."}
```

#### 3. `RunnablePassthrough`: pass the input through unchanged

`RunnablePassthrough()` forwards its input as-is. `RunnablePassthrough.assign(key=runnable)` keeps every existing key and **adds** a new one, which is the most useful pattern for multi-step pipelines.

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough.assign(report=writer_chain)
chain.invoke({"topic": "AI agents", "research": "..."})
# {"topic": "AI agents", "research": "...", "report": "# Introduction ..."}
```

#### 4. `RunnableLambda`: turn any Python function into a step

```python
from langchain_core.runnables import RunnableLambda

def extract_score(text: str) -> int:
    # "Score: 8/10" -> 8
    line = next(l for l in text.splitlines() if l.startswith("Score:"))
    return int(line.split(":")[1].split("/")[0])

score_chain = critic_chain | RunnableLambda(extract_score)
score_chain.invoke({"report": report})   # 8
```

Inside a chain, a plain function or lambda is converted automatically, so `critic_chain | extract_score` also works.

#### 5. `RunnableBranch`: choose a path based on the input

```python
from langchain_core.runnables import RunnableBranch

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# Rewrite the report only if the critic scored it below 7
maybe_rewrite = RunnableBranch(
    (lambda x: x["score"] < 7, rewrite_chain),
    RunnableLambda(lambda x: x["report"]),   # default: keep the report
)
```

For simple routing you can also return a Runnable from a `RunnableLambda`; LCEL runs whatever Runnable the function returns.

#### 6. Fallbacks and retries

```python
from langchain_openai import ChatOpenAI

primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
backup = ChatOpenAI(model="gpt-4o", temperature=0)

# Retry transient errors (rate limits, timeouts), then fall back to another model
robust_llm = primary.with_retry(stop_after_attempt=3).with_fallbacks([backup])

writer_chain = writer_prompt | robust_llm | StrOutputParser()
```

#### 7. Config: tags, metadata, concurrency

Every Runnable accepts a `config` dict. Tags and metadata appear in LangSmith traces.

```python
writer_chain.invoke(
    {"topic": topic, "research": research},
    config={"tags": ["writer"], "metadata": {"topic": topic}},
)
```

### Example: Writer + Critic as one LCEL chain

The pipeline currently calls the two chains separately. With `RunnablePassthrough.assign` they can be combined into a single Runnable that carries all keys forward:

```python
from langchain_core.runnables import RunnablePassthrough

write_and_review = (
    RunnablePassthrough.assign(report=writer_chain)      # adds "report"
    | RunnablePassthrough.assign(feedback=critic_chain)  # critic_prompt reads {report}
)

result = write_and_review.invoke({"topic": topic, "research": research_combined})
result["report"]     # the drafted report
result["feedback"]   # "Score: X/10 ..."
```

Data flow:

```
{topic, research}
   → assign(report)    → {topic, research, report}
   → assign(feedback)  → {topic, research, report, feedback}
```

Extra keys are fine: `critic_prompt` only reads `{report}` and ignores the rest.

### Streaming the report into Streamlit

Because `writer_chain` ends with `StrOutputParser`, `.stream()` yields plain text chunks, which Streamlit can render as they arrive:

```python
import streamlit as st

stream = writer_chain.stream({"topic": topic, "research": research_combined})
report = st.write_stream(stream)   # shows tokens live, returns the full text
```

### Batch: research several topics at once

```python
inputs = [
    {"topic": "Quantum computing", "research": r1},
    {"topic": "Solid-state batteries", "research": r2},
]
reports = writer_chain.batch(inputs, config={"max_concurrency": 2})
```

### Async version (for FastAPI or other async servers)

```python
report = await writer_chain.ainvoke({"topic": topic, "research": research})

async for chunk in writer_chain.astream({"topic": topic, "research": research}):
    print(chunk, end="", flush=True)
```

### Inspecting and debugging a chain

```python
writer_chain.input_schema.model_json_schema()   # required inputs: topic, research
writer_chain.get_graph().print_ascii()           # needs: pip install grandalf
writer_prompt.invoke({"topic": "x", "research": "y"}).to_messages()  # see the exact prompt
```

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to see every step, its inputs, outputs, and latency in LangSmith.

### Common mistakes

| Mistake | Fix |
|---------|-----|
| `KeyError` / missing variable in prompt | The dict passed to `invoke` must contain every `{placeholder}` in the prompt |
| Passing a `str` into a prompt that expects a dict | Wrap it: `{"report": RunnablePassthrough()} \| critic_prompt` |
| Getting an `AIMessage` instead of text | Add `StrOutputParser()` at the end |
| Using `LLMChain` from old tutorials | Replace with `prompt \| llm \| StrOutputParser()` |
| Putting model-driven decisions in a long LCEL chain | Use an agent (`create_agent`) or LangGraph instead |

### When to use what

| Use | When |
|-----|------|
| **LCEL chain** | The steps are fixed and known in advance (Writer, Critic) |
| **Agent (`create_agent`)** | The model must decide which tools to call (Search, Reader) |
| **LangGraph** | You need loops, shared state, or conditional flow, e.g. "rewrite until the critic score is ≥ 8" |

---

## 📊 Example Output

The system generates reports with:
- Comprehensive introduction and background
- Key findings with detailed explanations
- Well-sourced conclusions
- Structured sections and proper formatting
- Quality scores from 1-10

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Search powered by [Tavily](https://tavily.com)
- UI built with [Streamlit](https://streamlit.io/)
- Inspired by agentic AI research patterns

---

## 📧 Support

For support, open an issue on GitHub or contact the maintainers.

**Happy Researching! 🚀**