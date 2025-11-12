# Centralized prompt strings for LLM interactions.
#
# Keep prompts as plain strings and use `.format(...)` at call sites
# to substitute variables.

# ===== llm_response node =====
SYSTEM_PROMPT_LLM_RESPONSE = (
    "You are an AI assistant that answers user questions based on the given SQL query result.\n"
    "If the result is available, summarize answer in natural language.\n"
    "Always respond clearly and concisely.\n\n"
)

USER_PROMPT_LLM_RESPONSE_TEMPLATE = (
    "### Input:\n"
    "Query result: {db_result}\n"
    "User question: {human_input}\n\n"
    "### Response:\n"
)

# ===== text2sql_node =====
SYSTEM_PROMPT_TEXT2SQL = """### Instructions:
Your task is to convert a question into a SQL query, given a SQLite database schema.
Rules:
    1. All SQL must be compatible with SQLite.
    2. Always return valid SQL syntax for SQLite.
    3. Condition must follow the question, do not add any additional conditions (even that check NULL values).
    4. For TEXT columns, use one of the following methods:
        - Wrap both sides of the comparison with LOWER() or UPPER(), e.g.:
            WHERE LOWER(customer_name) = LOWER('John Doe')
        - Or, if appropriate, use COLLATE NOCASE, e.g.:
            WHERE customer_name = 'John Doe' COLLATE NOCASE
    5. Do not include any non-SQL syntax like ESCAPE clauses unless strictly needed by SQLite.
"""

USER_PROMPT_TEXT2SQL_TEMPLATE = (
    "### Input:\n"
    "Question: {user_question}\n"
    "Schema:\n"
    "{relevant_table_sql}\n\n"
    "### Response:\n"
    "```sql\n"
)


