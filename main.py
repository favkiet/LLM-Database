from src.graph import graph
from langchain_core.messages import HumanMessage
from src.graph import invoke_llm_database

if __name__ == "__main__":
    
    # user_query = "What is the phone number of Customer#000000001?"
    # user_query = "How many customers are in Brazil?"
    # user_query = "Please list the phone numbers of all the customers in the household segment and are in Brazil."
    user_query = ("List all customers with invoices over 500,000."
        "customers refer to c_name; invoices over 500,000 refers to o_totalprice > 500000; use DISTINCT to avoid duplicate customer names;")
# Select DISTINCT customer.c_name From customer JOIN orders on customer.c_custkey = orders.o_custkey WHERE orders.o_totalprice > 500000)
    #user_query = "List all customers with invoices over 500,000"
    
    # Chọn provider: "ollama" hoặc "openai"
    # Mặc định là "ollama" nếu không truyền tham số provider
    provider = "openai"  # Thay đổi thành "openai" để sử dụng OpenAI
    
    # Gọi với provider
    response = invoke_llm_database(user_query, provider=provider)