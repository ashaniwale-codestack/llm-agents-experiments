#%%
import os
import json
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_API_KEY")
)

CUSTOMER_DB = {
    "CUST-441": {
        "name": "Asha Niwale",
        "average_transaction": 120,
        "max_ever_sent": 500,
        "countries_used": ["UK", "France"],
        "large_transfers": 0,
        "account_age_days": 842
    },
    "CUST-123": {
        "name": "John Smith",
        "average_transaction": 5000,
        "max_ever_sent": 15000,
        "countries_used": ["UK", "USA", "Nigeria", "UAE"],
        "large_transfers": 12,
        "account_age_days": 2100
    }
}

BLOCKED_CARDS=[]
#%%
@tool
def get_fraud_score(
    amount: float,
    location: str,
    time: str,
    is_new_device: bool
) -> str:
    """
    Calculate fraud risk score for a transaction.
    Returns score 0-100 and list of risk flags.
    """
    score = 0
    flags = []

    # Amount risk
    if amount > 5000:
        score += 30
        flags.append("very_high_amount")
    elif amount > 1000:
        score += 15
        flags.append("high_amount")

    # Location risk
    high_risk_countries = ["nigeria", "ghana", "cameroon"]
    if any(c in location.lower() for c in high_risk_countries):
        score += 25
        flags.append("high_risk_country")

    def extract_hour(time_str: str) -> int:
        time_str = time_str.strip()
        
        # "2024-03-16 14:30:00" → "14:30:00"
        if " " in time_str:
            time_str = time_str.split(" ")[1]
        
        # "2024-03-16T14:30:00" → "14:30:00"
        if "T" in time_str:
            time_str = time_str.split("T")[1]
        
        # "14:30:00" or "14:30" → 14
        return int(time_str.split(":")[0])

    hour = extract_hour(time)

    # Time risk
    print(f"time: {time}")
    print(f"hour: {hour}")
    # hour = int(time.split(":")[0])
    if hour < 6 or hour > 23:
        score += 20
        flags.append("unusual_time")

    # Device risk
    if is_new_device:
        score += 15
        flags.append("new_device")

    risk_level = (
        "HIGH"   if score >= 60 else
        "MEDIUM" if score >= 30 else
        "LOW"
    )

    return json.dumps({
        "score": score,
        "risk_level": risk_level,
        "flags": flags
    })

@tool
def get_customer_history(customer_id:str) -> str:
    """
    Retrieve customer's transaction history from database.
    """
    customer = CUSTOMER_DB.get(customer_id)

    if not customer:
        return json.dumps({"error": "Customer not found"})

    return json.dumps(customer)


@tool
def block_card(customer_id: str, reason: str) -> str:
    """
    Block customer's card and flag account for review.
    """
    BLOCKED_CARDS.append(customer_id)
    return json.dumps({
        "status": "blocked",
        "customer_id": customer_id,
        "reason": reason,
        "blocked_at": "03:14:22",
        "review_team_notified": True
    })


@tool
def approve_transaction(
    transaction_id: str,
    amount: float
) -> str:
    """
    Approve a transaction as legitimate.
    """
    return json.dumps({
        "status": "approved",
        "transaction_id": transaction_id,
        "amount": amount
    })

@tool
def flag_for_review(
    transaction_id: str,
    reason: str
) -> str:
    """
    Flag transaction for manual review by fraud team.
    Keep transaction pending until reviewed.
    """
    return json.dumps({
        "status": "pending_review",
        "transaction_id": transaction_id,
        "reason": reason,
        "estimated_review_time": "2 hours"
    })

#%%
agent = create_react_agent(
    llm, tools=[
        get_fraud_score,
        get_customer_history,
        block_card,
        approve_transaction,
        flag_for_review
    ],
    prompt ="""
    You are a bank fraud detection AI agent.
    when you recieve a transaction, follow the logic:
    - ALWAYS call get_fraud_score first
    - If score < 30  → approve immediately
    - If score 30-59 → get_customer_history, then decide:
        - History looks normal → flag_for_review
        - History looks suspicious → block_card + notify_customer
    - If score >= 60 → get_customer_history, then:
        - Almost always block_card + notify_customer
        - Exception: customer has history of similar transactions → flag_for_review

Always explain your reasoning clearly.
Always notify the customer of any action taken.
    """
)
#%%
def analyse_transaction(transaction:dict):
    """
        This is where transaction data enters the system.
        We convert the dict into a natural language message for the agent.
    """
    try:
        # Convert transaction dict → natural language message
        message = f"""
                Transaction ID: {transaction['id']}
                Customer ID:    {transaction['customer_id']}
                Amount:         £{transaction['amount']}
                Merchant:       {transaction['merchant']}
                Location:       {transaction['location']}
                Time (HH:MM only): {transaction['time']}
                New Device:     {transaction['new_device']}
IMPORTANT: When calling get_fraud_score, use ONLY the HH:MM 
value for the time parameter — not the full date
                """
        
        # result = agent.invoke({
        #     "messages": [("user", message)]
        # })
        # print(f"response: {result["messages"][-1].content}")

        for chunk in agent.stream(
            {"messages":[("user",message)]},
            stream_mode="updates"
        ):
            for node, values in chunk.items():
                if node == "tools":
                    for msg in values["messages"]:
                        print(f"\n Tool: {msg.name}")
                        print(f"   Result: {msg.content[:100]}...")
                elif node == "agent":
                    msg = values["messages"][-1]
                    if msg.content:
                        print(f"\n Final Decision:\n{msg.content}")


    except Exception as e:
        print(f"error: {e}")

#%%
# Transaction 1 — clearly fraudulent
analyse_transaction({
    "id": "TXN-9291",
    "customer_id": "CUST-441",
    "amount": 9000,
    "merchant": "Western Union",
    "location": "Nigeria",
    "time": "03:14",
    "new_device": True
})
# %%
