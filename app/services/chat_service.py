from flask import current_app
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, trim_messages
from typing import Sequence
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Sequence[BaseMessage]
    language: str

def initialize_model():
    """
    Initialize the OpenAI model and trimmer using Flask app configuration.
    """

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
    return model, trimmer

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and polite assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def call_model(state: State):
    """
    Call the OpenAI model with the given state and return the response.
    """

    model, trimmer = initialize_model()
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

async def run_workflow(messages: Sequence[HumanMessage], language: str):
    """
    Run the LangChain workflow with the provided messages and language.
    """
    
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "api-thread"}}
    output = await app.ainvoke({"messages": messages, "language": language}, config)
    return output["messages"][-1]