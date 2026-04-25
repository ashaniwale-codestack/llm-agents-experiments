#%%
from groq import Groq
import os
from dotenv import load_dotenv
import json
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-120b"
transactions = [
    {"amount": 100, "date": "22-04-2026","type": "debit"},
    {"amount": 530, "date": "24-04-2026","type": "debit"},
    {"amount": 320, "date": "24-04-2026","type": "credit"},
    {"amount": 1010, "date": "22-04-2026","type": "credit"},
    {"amount": 160, "date": "21-04-2026","type": "debit"},
]
#%%
tools=[
        {
            "type": "function",
            "function": {
                "name": "get_transactions_by_type",
                "description": "Get my all transactions by type.",
                "parameters": {
                    # JSON schema object
                    "type": "object",
                    "properties":{
                        "ctype":{
                            "type":"string",
                            "enum":["debit", "credit"]
                        }
                    },
                    "required":["ctype"]
                }
            }
        }
    ]

message = [{"role":"user", "content":"what are my all debit transactions of dated 22-04-2026" }]

def get_transactions_by_type(ctype):
    return [item for item in transactions if item["type"]==ctype]
#%%
response = client.chat.completions.create(
    model = model,
    messages = message,
    tools=tools,
    tool_choice="auto"
)

response_msg = response.choices[0].message
print(response_msg)

#%%
# 1. Check if the model wants to call a tool
tools_calls = response_msg.tool_calls
print(f"tools_calls: {tools_calls}")
available_functions = {
    "get_transactions_by_type": get_transactions_by_type
}
#%%
if tools_calls:
    for tool_call in tools_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)

    message.append(response_msg)
    message.append({
        "role": "tool",
        "content": str(function_response),
        "name": function_name,
        "tool_call_id": tool_call.id
    })

    final_response = client.chat.completions.create(
        model = model,
        messages = message
    )

    print(final_response.choices[0].message.content)


# %%
