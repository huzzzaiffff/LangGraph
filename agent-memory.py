from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver     # NEW

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

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# --- Node ---
def assistant(state: MessagesState):
    print("   [assistant running, messages in state:", len(state["messages"]), "]")
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# --- Graph (same as before) ---
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Version 1: NO memory
react_graph = builder.compile()

# Version 2: WITH memory (NEW)
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

def show(result):
    for m in result["messages"]:
        m.pretty_print()

# ========== PART A: no memory ==========
print("\n##### A1: no memory, 'Add 3 and 4.' #####")
r = react_graph.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]})
show(r)

print("\n##### A2: no memory, 'Multiply that by 2.' #####")
r = react_graph.invoke({"messages": [HumanMessage(content="Multiply that by 2.")]})
show(r)

# ========== PART B: with memory, thread "1" ==========
config = {"configurable": {"thread_id": "1"}}

print("\n##### B1: memory, thread 1, 'Add 3 and 4.' #####")
r = react_graph_memory.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]}, config)
show(r)

print("\n##### B2: memory, thread 1, 'Multiply that by 2.' #####")
r = react_graph_memory.invoke({"messages": [HumanMessage(content="Multiply that by 2.")]}, config)
show(r)

# ========== PART C: a different thread ==========
config2 = {"configurable": {"thread_id": "2"}}

print("\n##### C1: memory, thread 2 (fresh), 'Multiply that by 2.' #####")
r = react_graph_memory.invoke({"messages": [HumanMessage(content="Multiply that by 2.")]}, config2)
show(r)

# ========== PART D: peek at what is saved ==========
saved = react_graph_memory.get_state(config)
print("\n##### D: saved messages in thread 1:", len(saved.values["messages"]), "#####")