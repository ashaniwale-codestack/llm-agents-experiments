#%%
from groq import Groq
import json
from dotenv import load_dotenv
import os
load_dotenv()

#%%

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-120b"

groceries = [
    {"name": "onion", "quantity": 1, "purchaseDate": "22-04-2026", "type": "veggies"},
    {"name": "banana", "quantity": 6, "purchaseDate": "21-04-2026", "type": "fruits"},
    {"name": "apple", "quantity": 4, "purchaseDate": "21-04-2026", "type": "fruits"},
    {"name": "milk", "quantity": 1, "purchaseDate": "21-04-2026", "type": "dairy"},
]

def get_groceries_by_date(date):
    """"Get all groiceries from my house"""
    return [item for item in groceries if item["purchaseDate"]==date]

def get_groceries_by_type(gtype:str):
    """"Get all groiceries by type"""
    return [item for item in groceries if item["type"]==gtype]

def check_missing_category(items:list):
    categories = {"fruits", "veggies", "dairy"}
    present = {g["type"] for g in items}
    missing = list(categories - present)

    if missing:
        return {"restock_needed": True, "missing": missing}
    return {"restock_needed": False}
    
# %% function test
print(get_groceries_by_date("22-04-2026"))


# %%
tools = [
    {
        "type": "function",
        "function":{
            "name": "get_groceries_by_date",
                "description": "Get all groceries by date",
                "parameters": {
                    # JSON schema object
                    "type": "object",
                    "properties":{
                        "date":{
                            "type":"string",
                            "description": "Grocery purchase date, e.g. 22-04-2026"
                        }
                    },
                    "required":["date"]
                }
        }
    },    
    {
        "type": "function",
        "function":{
            "name": "get_groceries_by_type",
                "description": "Filter groceries by type",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "gtype":{
                            "type":"string",
                            "description": "Grocery Type, e.g. fruits"
                        }
                    },
                    "required":["gtype"]
                }
        }
    },    
    {
        "type": "function",
        "function":{
            "name": "check_missing_category",
                "description": "Check if any grocery categories are missing",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "items":{
                            "type":"array"
                        }
                    },
                    "required":["items"]
                }
        }
    }
]

#%%
message =[
    {
        "role":"system", "content":"""You are a kitchen assistant that helps users query grocery data.
                        You have access to tools to fetch and analyze grocery information.

                        Rules:
                        - Always use tools when data is required (do not guess).
                        - First, retrieve groceries using the appropriate tool (e.g., by date or type).
                        - Then, if needed, analyze the tool results (e.g., filtering or checking missing categories).
                        - After receiving tool results, provide a clear final answer to the user.
                        - Do NOT call tools in the final response step. Only use tools when needed to get data.
                        - Never return raw tool output; always summarize it in natural language.
                        After tool results are provided, DO NOT call any more tools.
                        Use the provided data to directly answer the user.
                        """
    },
    {
        "role":"user", "content":"""Give me a list of all groceries purchased on 22-04-2026 and tell me which items are missing."""
    }
]


response = client.chat.completions.create(
    model = model,
    messages = message,
    tools = tools,
    temperature=0.5,
    tool_choice="auto"
)

response_msg = response.choices[0].message

# %%
tool_calls = response_msg.tool_calls or []
print(f"tool_calls: {tool_calls}")

# %%
message.append(response_msg)

available_functions = {
    "get_groceries_by_date": get_groceries_by_date,
    "get_groceries_by_type":get_groceries_by_type,
    "check_missing_category":check_missing_category
}

for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(**function_args)

    message.append(
        {
            "role": "tool",
            "content": str(function_response),
            "tool_call_id": tool_call.id,
        }
    )

# The loop continues, sending the tool results back to the model
while True:
    # 1. Send the current message history to the model
    response = client.chat.completions.create(
        model=model, # Use a supported Groq model
        messages=message,
        tools=tools,
        tool_choice="auto"
    )
    
    response_msg = response.choices[0].message
    
    # 2. Check: Did the model return text (final answer) or a tool call?
    if response_msg.content:
        print(f"Final response: {response_msg.content}")
        break  # We are done!

    if response_msg.tool_calls:
        # Append the assistant's request to call tools
        message.append(response_msg)
        
        # 3. Execute all requested tools
        for tool_call in response_msg.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Executing: {function_name}...")
            result = function_to_call(**function_args)

            # Append the result so the model can see it in the next loop
            message.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
# %%
