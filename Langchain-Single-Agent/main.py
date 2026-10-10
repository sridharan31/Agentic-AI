import os
import certifi


import requests

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain import hub

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ==========================================
# SEARCH TOOL
# ==========================================

if TAVILY_API_KEY:
    search_tool = TavilySearchResults(
        api_wrapper=TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY),
        max_results=2
    )
else:
    @tool
    def search_tool(query: str) -> str:
        """Search the web for information."""
        return "Web search is unavailable because TAVILY_API_KEY is not configured."

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ==========================================
# LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# ==========================================
# PROMPT
# ==========================================

prompt = hub.pull("hwchase17/react")

# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ==========================================
# RUN
# ==========================================

response = agent_executor.invoke({
    "input": (
        "Find the capital of India"
        "and then find its current weather."
    )
})

print("\n========================")
print("FINAL OUTPUT")
print("========================\n")

print(response["output"])