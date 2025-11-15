from src.graph import graph
from langchain_core.messages import HumanMessage
from src.graph import invoke_llm_database

if __name__ == "__main__":
    
    # user_query = "What is the phone number of Customer#000000001?"
    # user_query = "How many customers are in Brazil?"
    user_query = "Please list the phone numbers of all the customers in the household segment and are in Brazil."
    # user_query = "How many total customers are there?"
    # user_query = "List all customers with invoices over 500,000"
    response = invoke_llm_database(user_query)