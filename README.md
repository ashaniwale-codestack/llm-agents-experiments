# LLM Agent System with Tool Calling & LangChain Workflows (Groq + Python)

A hands-on project for building and experimenting with **LLM-powered agent systems** using tool calling, multi-step reasoning, and LangChain-inspired workflows.

This project demonstrates how modern AI applications dynamically decide when and how to use external tools to solve real-world problems.

---

## 🚀 Capabilities

- Dynamic tool selection (function calling)
- Multi-step reasoning pipelines
- LLM agent workflow design (LangChain-inspired)
- Fast inference using Groq
- Extensible tool/function architecture

---

## 🧠 What This Project Does

This system builds an **AI agent that can reason and act using tools**.

Example workflow:

User asks:
> "What groceries did I buy on 22-04-2026 and what am I missing?"

The agent:
1. Calls a function to fetch groceries by date  
2. Filters and processes the data  
3. Checks for missing categories (fruits, veggies, dairy)  
4. Returns a final natural language response  

---

## 📁 Project Structure

```

llm-agents-experiments/
├── groq-functions/
│   └── groq_multi-function_calling.py
├── requirement.txt
└── README.md

````

---

## ⚙️ Setup

### Prerequisites
- Python 3.8+
- Groq API key

### Installation

```bash
git clone <repo-url>
cd langchain-experiments
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
pip install -r requirement.txt
````

### Environment Variables

Create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

---

## 🧪 Example Use Case

### Grocery AI Agent

User input:

> "Give me groceries purchased on 22-04-2026 and tell me what I am missing"

### Execution Flow:

* Step 1: Get groceries by date
* Step 2: Filter and analyze data
* Step 3: Detect missing categories
* Step 4: Return final response

### Output:

> You purchased onions and bananas on 22-04-2026. You are missing dairy items.

---

## 🔧 Technologies Used

* Groq API
* LangChain-inspired agent design
* Python
* Function Calling (Tools / Agents)

---

## 📦 Dependencies - **Groq**: Fast LLM API for inference - **python-dotenv**: Environment variable management

---

## 🎯 Learning Outcomes

* How LLMs use external tools
* Building multi-step reasoning systems
* Designing agent-based architectures
* Practical function-calling implementation

---

## 📌 Notes

- All API keys should be stored in .env file (never commit to version control) 
- The .env file is included in .gitignore for security 
- Examples use realistic grocery management as a demonstration domain

---

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/) 
- [Groq Documentation](https://console.groq.com/docs) 
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

```
**Last Updated**: April 2026