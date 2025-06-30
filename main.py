from rich import print

from typing import Annotated, Dict, Any, Iterator, Tuple
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage

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

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})


def main():
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    # Create tool node that wraps available tools in a ToolNode container
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # The `tools_condition` function returns "tools" if the chatbot asks to
    # use a tool, and "END" if it is fine directly responding. This conditional
    # routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
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

    # Create a configuration
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(graph, human_command, config, user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, human_command, config, user_input)
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
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


def stream_graph_updates(
    graph: CompiledStateGraph,
    human_command: Command,
    config: RunnableConfig | None,
    user_input: str,
) -> None:
    """
    Stream and print updates from the LangGraph compiled graph based on user input.
    """
    input_message = [{"role": "user", "content": user_input}]

    inputs_to_stream = [
        {"messages": input_message},  # User input
        human_command,  # Human command
    ]

    # Stream both inputs using the same pattern
    for input_data in inputs_to_stream:
        assistant_started = False  # Track if we've started printing assistant output
        stream_results = graph.stream(
            input_data, config=config, stream_mode="messages"
        )

        for step, metadata in stream_results:
            node_name = metadata["langgraph_node"]
            
            # Handle chatbot responses
            if node_name == "chatbot":
                if isinstance(step, AIMessage):
                    # Print tool calls if any
                    if hasattr(step, "tool_calls") and step.tool_calls:
                        for tool_call in step.tool_calls:
                            print(f"🔍 Calling tool: {tool_call['name']}")
                            print(f"   Arguments: {tool_call['args']}")
                    
                    # Print assistant text response
                    if hasattr(step, "text") and step.text():
                        if not assistant_started:
                            print("\n🤖 Assistant: ", end="\n\n")
                            assistant_started = True
                        print(step.text(), end="")
            
            # Handle tool responses
            elif node_name == "tools":
                print()
                if isinstance(step, ToolMessage):
                    tool_call_id = getattr(step, "tool_call_id", "Unknown")
                    print(f"🔧 Tool Result (ID: {tool_call_id}): {step.content}")
                elif hasattr(step, "content"):
                    print(f"🔧 Tool Output: {step.content}")
        
        print()
        print("-" * 50)  # Separator between iterations


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
