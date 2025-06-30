from rich import print

from typing import Annotated, Dict, Union, Any, cast
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
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
    name: str
    birthday: str


graph_builder = StateGraph(State)


# Create tools
@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Request assistance from a human.
    
    :param name: name of an entity
    :type name: str
    :param birthday: birthday of an entity
    :type birthday: str
    :param tool_call_id: ID of the tool call
    :type tool_call_id: InjectedToolCallId
    :return: Command object
    :rtype: Command
    """
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


tool = TavilySearch(max_results=2)
tools: list = [tool, human_assistance]

# Create a chat model
llm = init_chat_model("qwen3", model_provider="ollama")
llm_with_tools = llm.bind_tools(tools)

human_response = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

human_command: Command = Command(resume={"data": human_response})

# Can you look up when LangGraph was released? When you have the answer, use the human_assistance tool for review

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
    assert isinstance(message, AIMessage)
    assert (len(message.tool_calls) <= 1)
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
            # Proper type annotation for metadata
            if not isinstance(metadata, dict):
                continue
            
            node_name = metadata.get("langgraph_node", "")
            
            # Handle chatbot responses
            if node_name == "chatbot":
                if isinstance(step, AIMessage):
                    # Print tool calls if any
                    if hasattr(step, "tool_calls") and step.tool_calls:
                        for tool_call in step.tool_calls:
                            # Ensure tool_call is a dictionary-like object
                            if isinstance(tool_call, dict):
                                print(f"🔍 Calling tool: {tool_call.get('name', 'Unknown')}")
                                print(f"   Arguments: {tool_call.get('args', {})}")
                            else:
                                # Handle case where tool_call has attributes instead of dict access
                                tool_name = getattr(tool_call, 'name', 'Unknown')
                                tool_args = getattr(tool_call, 'args', {})
                                print(f"🔍 Calling tool: {tool_name}")
                                print(f"   Arguments: {tool_args}")
                    
                    # Print assistant text response
                    if hasattr(step, "content") and step.content:
                        if not assistant_started:
                            print("\n🤖 Assistant: ", end="\n\n")
                            assistant_started = True
                        print(step.content, end="")
            
            # Handle tool responses
            elif node_name == "tools":
                print()
                if isinstance(step, ToolMessage):
                    tool_call_id = getattr(step, "tool_call_id", "Unknown")
                    print(f"🔧 Tool Result (ID: {tool_call_id}): {step.content}")
                elif isinstance(step, BaseMessage) and hasattr(step, "content"):
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
