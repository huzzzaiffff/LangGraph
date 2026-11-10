from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))          # NEW: the node that runs the tool

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)  # NEW: route by the AI's output
builder.add_edge("tools", END)

graph = builder.compile()

# Run A: no tool needed
out = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
for m in out["messages"]:
    m.pretty_print()

# Run B: tool needed
out = graph.invoke({"messages": [HumanMessage(content="What is 23 multiplied by 3?")]})
for m in out["messages"]:
    m.pretty_print()