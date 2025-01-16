import os
from getpass import getpass
import gradio as gr

# Install necessary libraries
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, MessagesState, StateGraph
except ImportError:
    print("Make sure you have installed langchain-core, langgraph, and langchain-openai.")
    print("Use: pip install langchain-core langgraph>0.2.27 langchain-openai")
    exit()

# Set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini")

# Define a prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="messages"),
])

# Define a trimmer to manage conversation history
trimmer = trim_messages(
    max_tokens=100, strategy="last", token_counter=model, include_system=True
)

# Define the chatbot workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    # Trim messages to manage conversation history
    trimmed_messages = trimmer.invoke(state["messages"])
    # Use the prompt template
    prompt = prompt_template.invoke({"messages": trimmed_messages, "language": state.get("language", "English")})
    # Get response from the model
    response = model.invoke(prompt)
    return {"messages": [response]}

# Add the model invocation as a node in the workflow
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Set up in-memory persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Function to handle Gradio inputs and outputs
def chatbot_interface(user_input):
    input_messages = [HumanMessage(content=user_input)]
    config = {"configurable": {"thread_id": "chat_session"}}
    output = app.invoke({"messages": input_messages, "language": "English"}, config)
    response = output["messages"][-1]
    return response.content

# Create Gradio interface
iface = gr.Interface(fn=chatbot_interface, inputs="text", outputs="text", title="LangChain Chatbot")

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()