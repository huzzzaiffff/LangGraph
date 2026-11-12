from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# --- Tools ---
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# --- Model ---
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant that does arithmetic using the provided tools, one tool call at a time.")

# --- Node ---
def assistant(state: MessagesState):
    print("--- assistant running, messages so far:", len(state["messages"]))
    reply = llm_with_tools.invoke([sys_msg] + state["messages"])
    print("    tool_calls:", reply.tool_calls)
    print("    content:", repr(reply.content[:100]))
    return {"messages": [reply]}

# --- Graph ---
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")      # THE LOOP

react_graph = builder.compile()

# --- Run ---
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
result = react_graph.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
