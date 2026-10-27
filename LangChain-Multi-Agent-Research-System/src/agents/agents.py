from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# Model Initialization
# Meta Model API (dev.meta.ai) is OpenAI-compatible, so ChatOpenAI works with its base_url.
# Priority: local Ollama (OLLAMA_MODEL) -> Meta (MODEL_API_KEY) -> OpenAI.
if os.getenv("OLLAMA_MODEL"):
    from langchain_ollama import ChatOllama
    # num_ctx raised so the combined search + scraped research fits in the prompt
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0, num_ctx=8192)
elif os.getenv("MODEL_API_KEY"):
    llm = ChatOpenAI(
        model=os.getenv("META_MODEL", "muse-spark-1.3"),
        base_url="https://api.meta.ai/v1",
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=0,
    )
else:
    llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)


# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
       
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url],

    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

