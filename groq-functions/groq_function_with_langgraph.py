#%%
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# %%
import os
import langchain
print(f"Version: {langchain.__version__}")
print(f"Location: {os.path.dirname(langchain.__file__)}")

# %%
model = "openai/gpt-oss-120b"
transactions = [
    {"amount": 100, "date": "22-04-2026","type": "debit"},
    {"amount": 530, "date": "24-04-2026","type": "debit"},
    {"amount": 320, "date": "24-04-2026","type": "credit"},
    {"amount": 1010, "date": "22-04-2026","type": "credit"},
    {"amount": 160, "date": "21-04-2026","type": "debit"},
]

@tool
def get_transactions_by_type(ctype:str) -> str:
    """Check the transactions by type"""
    return [item for item in transactions if item["type"]==ctype]

llm = ChatGroq(
    model = model,
    api_key =os.getenv("GROQ_API_KEY")
)

agent = create_react_agent(
    llm, tools=[
        get_transactions_by_type
    ]
)

# run
result = agent.invoke({
    "messages": [("user","what are my all debit & credit transactions of dated 22-04-2026 and show reaminig balance.")]
 })

print(result["messages"][-1].content)
# %%
