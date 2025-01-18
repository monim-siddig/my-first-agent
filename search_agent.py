from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from IPython.display import display

os.environ["LANGSMITH_TRACING"]="true"

tavily_search = TavilySearchResults(max_results=2)

#tools = [search]
tools = [tavily_search]

search_tool = ToolNode(tools)

#model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(tools)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["search_tool", END]:
    messages = state['messages']
    last_message = messages[-1]
    #print(last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "search_tool"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
graph_builder = StateGraph(MessagesState)

# Define the two nodes we will cycle between
graph_builder.add_node("agent", call_model)
graph_builder.add_node("search_tool", search_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
graph_builder.add_edge(START, "agent")

# We now add a conditional edge
graph_builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `search_tools` to `agent`.
# This means that after `search_tools` is called, `agent` node is called next.
graph_builder.add_edge("search_tool", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = graph_builder.compile(checkpointer=checkpointer)

display(app.get_graph().draw_ascii()) 
        
# Use the Runnable
humban_message = HumanMessage(content="What is the weather is sf")
#user_input = input("Hi, how can I help you? ")
#prompt_message = HumanMessage(content=user_input)

#pp = f"""search for the query delimeted by three backticks below and put the response in a nice html format
#```{user_input}```"""
#pp_mes = HumanMessage(content=pp)
#final_state = app.invoke(
#    {"messages": [pp_mes]},
#    config={"configurable": {"thread_id": 42}}
#)

#agent_resp = final_state["messages"][-1].content

#print(f'\n\n-------\n{agent_resp}\\n\n-------------\n\n')

#display(HTML(agent_resp))