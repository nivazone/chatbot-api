from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import trim_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

async def main():
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "abc789"}}
    query = "Hi! I'm Todd."
    language = "English"

    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages, "language": language}, config)
    output["messages"][-1].pretty_print()

    query = "What's my name?"

    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages, "language": language}, config)
    output["messages"][-1].pretty_print()


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

trimmer = trim_messages(
    max_tokens=65,
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
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

asyncio.run(main())
