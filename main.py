from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph, MessagesState

from langgraph.prebuilt import ToolNode, tools_condition
from pdf_search import search_pdf

from IPython.display import display, HTML


graph_builder = StateGraph(MessagesState)


search_tavily = TavilySearchResults(max_results=2)
tools = [search_tavily, search_pdf]
tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


def ai_agent(state: MessagesState):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


graph_builder.add_node("ai_agent", ai_agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "ai_agent")
graph_builder.add_edge("tools", "ai_agent")

graph_builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "ai_agent",
    # Next, we pass in the function that will determine which node is called next.
    tools_condition,
    )

# Any time a tool is called, we return to the chatbot to decide the next step

graph_builder.add_edge("tools", "ai_agent",)
graph = graph_builder.compile()

gg = graph.get_graph()
print(gg)
display(gg.draw_ascii())
"""'ai_agent': {'messages': [AIMessage(content='I"""
                                    
def stream_graph_updates(user_input: str):
    messages = []
    for event in graph.stream({"messages": [("user", user_input)]}):
        #print(f"{event}\n\n ------------\n\n")
        for value in event.values():
            messages.append(value)
            #print("Assistant:", value["messages"][-1].content)
    return messages


file = "docs/git_tutorial.pdf"
query = "What is Git"

system_message = f"""Follow the instructions below IN ORDER, AND DO NOT SCIP ANY STEP of them:
    1- use pdf_search tool to search the file {file} for the query delimeted by three hashes below.
    2- Summerize the response from pdf_search into a variable called PDF_RESULT.
    3- use the tavily_search tool, to search the web for the same query below AND summerize the response 
	    into a variable called WEB_RESULT.
    4- Combine and enhance the results from PDF_RESULT and WEB_RESULT into a single variable called \
        COMBINED_RESULT that is at most three paragraphs.
    5- your final output should only be a nicly formatted text in the form of:
        pdf search result:<PDF_RESULT>
        web search result: <WEB_RESULT>
        Conclusion:<COMBINED_RESULT>    
    the query:###{query}###"""

print(f'{system_message}\n\n ------------ \n\n')
response = stream_graph_updates(system_message)[-1]

output = response['messages'][-1].content
print(f"{response['messages'][-1].content} \n\n\n")

#resp = llm.invoke(chat_prompt)
#print(resp.content)
