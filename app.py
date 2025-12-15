import streamlit as st
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Page config
st.set_page_config(
    page_title="Travel Budget Planner",
    page_icon="globe",
    layout="centered"
)

st.title("Travel Budget Planner")
st.markdown("Plan your travel budget using AI-powered tools")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Hugging Face API Key", type="password", help="Enter your Hugging Face API key")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    st.markdown("---")
    st.markdown("### MCP Server Status")
    server_url = st.text_input("Server URL", value="http://localhost:3333/sse")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False
if "tools" not in st.session_state:
    st.session_state.tools = None
if "agent" not in st.session_state:
    st.session_state.agent = None

async def initialize_agent(server_url: str):
    """Initialize MCP client and create LangChain agent."""
    try:
        # Create MCP client
        client = MultiServerMCPClient(
            {
                "budget": {
                    "url": server_url,
                    "transport": "sse",
                }
            }
        )
        
        # Get tools from MCP server
        tools = await client.get_tools()
        
        # Create LLM using Hugging Face with ChatHuggingFace wrapper for tool support
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)
        
        # Create the LangChain Agent using langgraph
        agent = create_react_agent(llm, tools)
        
        return tools, agent, None
    except Exception as e:
        return None, None, str(e)

async def run_agent_async(agent, user_request: str):
    """Run the agent with user request asynchronously."""
    try:
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_request}]})
        # Extract the last message content
        if response.get("messages"):
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                return last_message.content
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

def run_agent(agent, user_request: str):
    """Run the agent with user request."""
    try:
        response = asyncio.run(run_agent_async(agent, user_request))
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Connect button
if st.sidebar.button("Connect to MCP Server"):
    if not api_key:
        st.sidebar.error("Please enter your OpenAI API key first!")
    else:
        with st.spinner("Connecting to MCP server..."):
            tools, agent, error = asyncio.run(initialize_agent(server_url))
            if error:
                st.sidebar.error(f"Connection failed: {error}")
            else:
                st.session_state.tools = tools
                st.session_state.agent = agent
                st.session_state.agent_initialized = True
                st.sidebar.success(f"Connected! Found {len(tools)} tool(s)")

# Show connection status
if st.session_state.agent_initialized:
    st.success("Connected to MCP Server")
    if st.session_state.tools:
        with st.expander("Available Tools"):
            for tool in st.session_state.tools:
                st.write(f"**{tool.name}**: {tool.description}")
else:
    st.warning("Not connected. Enter your API key and click 'Connect to MCP Server' in the sidebar.")

# Chat interface
st.markdown("---")
st.subheader("Chat with Travel Agent")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about travel budgets (e.g., 'Estimate budget for 5 days in Paris')"):
    if not st.session_state.agent_initialized:
        st.error("Please connect to the MCP server first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_agent(st.session_state.agent, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Quick actions
st.markdown("---")
st.subheader("Quick Budget Estimates")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Paris - 5 days"):
        if st.session_state.agent_initialized:
            st.session_state.messages.append({"role": "user", "content": "Estimate budget for 5 days in Paris"})
            response = run_agent(st.session_state.agent, "Estimate budget for 5 days in Paris")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

with col2:
    if st.button("Tokyo - 7 days"):
        if st.session_state.agent_initialized:
            st.session_state.messages.append({"role": "user", "content": "Estimate budget for 7 days in Tokyo"})
            response = run_agent(st.session_state.agent, "Estimate budget for 7 days in Tokyo")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

with col3:
    if st.button("New York - 3 days"):
        if st.session_state.agent_initialized:
            st.session_state.messages.append({"role": "user", "content": "Estimate budget for 3 days in New York"})
            response = run_agent(st.session_state.agent, "Estimate budget for 3 days in New York")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by MCP + LangChain + Hugging Face")
