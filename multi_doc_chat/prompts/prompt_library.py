from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt to rewrite user queries as standalone questions
# This helps in multi-turn conversations where the user might use pronouns or incomplete references.
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),  # Placeholder for previous conversation messages
    ("human", "{input}"),  # Placeholder for the latest user input
])

# Prompt to answer questions based on retrieved context
# The assistant must rely only on provided context and answer concisely (max 3 sentences).
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),  # Include prior messages for context
    ("human", "{input}"),  # Latest user query
])

# Central dictionary to register all prompts for easy access elsewhere in the project
PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}

## Example usage:
## 1. "Hello there, I want to study about RAG"
## 2. "What is the full form of it"   # This would be rewritten as "What is the full form of RAG?" by contextualize_question_prompt
