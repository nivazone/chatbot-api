from flask import current_app
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, trim_messages
from typing import Sequence, Dict
from typing_extensions import TypedDict

model = None
trimmer = None
# Global dictionary to store conversation state per chat_id
conversation_states: Dict[str, Sequence[BaseMessage]] = {}

class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str

def initialize_chat_service():

    print("Initializing chat service...")

    """
    Initialize the OpenAI model and trimmer using Flask app configuration.
    Called once during application startup.
    """
    global model, trimmer

    model_name = current_app.config.get("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens = current_app.config.get("MAX_TOKENS", 1000)

    model = ChatOpenAI(model=model_name)
    trimmer = trim_messages(
        max_tokens=max_tokens,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant but a real bogan. Answer all questions to the best of your ability in {language}, but your answers should mimic how a bogan would speak.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def call_model(state: State):
    """
    Call the OpenAI model with the given state and return the response.
    """

    global model, trimmer

    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

async def run_workflow(messages: Sequence[HumanMessage], language: str, chat_id: str):
    """
    Run the LangChain workflow with the provided messages and language.
    """

    global conversation_states

    # Load existing conversation history for the chat_id
    conversation_history = conversation_states.get(chat_id, [])

    # Add the new messages to the conversation history
    conversation_history.extend(messages)

    # Prepare state with the updated conversation history
    state = {"messages": conversation_history, "language": language}
    
    # Run the LangChain workflow
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "api-thread-{chat_id}"}}
    # output = await app.ainvoke({"messages": messages, "language": language}, config)
    output = await app.ainvoke(state, config)

    # Update the conversation state with the latest response
    conversation_history.append(output["messages"][-1])
    conversation_states[chat_id] = conversation_history

    return output["messages"][-1]