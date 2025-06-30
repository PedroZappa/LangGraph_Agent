import json
from rich import print

from typing import Annotated, Dict, Any, Iterator, Tuple
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# Create tools
@tool
def human_assistance(query: str) -> str:
    """
    Request assistance from a human.

    :param: query: The query to ask the human
    :type: str
    :return: The response from the human
    :rtype: str
    """
    human_response = interrupt({"query": query})
    return human_response["data"]


tool = TavilySearch(max_results=2)
tools: list = [tool, human_assistance]

# Create a chat model
llm = init_chat_model("qwen3", model_provider="ollama")
llm_with_tools = llm.bind_tools(tools)


def main():
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    # Create tool node that wraps available tools in a ToolNode container
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # The `tools_condition` function returns "tools" if the chatbot asks to 
    # use a tool, and "END" if it is fine directly responding. This conditional 
    # routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the 
        # condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Creates a direct edge from "tools" back to "chatbot"
    # After any tool is executed, control always returns to the chatbot node
    graph_builder.add_edge("tools", "chatbot")
    # Any time a tool is called, we return to the chatbot to decide the next step
    # Add an entry point to tell the graph where to start its work each time it is run
    graph_builder.add_edge(START, "chatbot")

    # Get memory
    memory = MemorySaver()
    # Compile the graph
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(graph, config, user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, config, user_input)
            break

    return 0


def chatbot(state: State) -> dict:
    """
    Process user messages through an LLM with tool capabilities. 

    :param state: The state of the graph
    :type state: State
    :return: The response from the LLM
    :rtype: dict
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig | None, user_input: str
) -> None:
    """
    Stream and print updates from the LangGraph compiled graph based on user input.

    :param graph: The compiled LangGraph graph to execute
    :type graph: CompiledStateGraph
    :param config: Optional configuration for the graph execution
    :type config: RunnableConfig | None
    :param user_input: The user's input message to process
    :type user_input: str
    :return: This function prints output directly and does not return a value
    :rtype: None
    """
    input_message = {"role": "user", "content": user_input}

    # Explicitly type the stream results
    stream_results: Iterator[Dict[Any, Dict[str, Any]]] = graph.stream(
        {"messages": [input_message]}, config=config, stream_mode="messages"
    )

    for step, metadata in stream_results:
        if metadata["langgraph_node"] == "chatbot" and hasattr(step, "text"):
            if text := step.text():
                print(text, end="")
    print()


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.

    :param state: The state of the graph
    :type state: State
    :return: The node to route to
    :rtype: str
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


if __name__ == "__main__":
    main()
