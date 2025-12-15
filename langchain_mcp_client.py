import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set your Hugging Face API key
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-hf-token-here"

# Global variables for tools and agent
tools = None
agent = None

async def initialize_agent():
    """Initialize MCP client and create LangChain agent."""
    global tools, agent
    
    # Step 1: Create MCP client that connects to budget server via SSE
    client = MultiServerMCPClient(
        {
            "budget": {
                "url": "http://localhost:3333/sse",
                "transport": "sse",
            }
        }
    )
    
    # Step 2: Get tools from MCP server
    tools = await client.get_tools()
    print("[OK] Connected to MCP Server!")
    print(f"Available tools: {[tool.name for tool in tools]}\n")
    
    # Step 3: Create LLM using Hugging Face with ChatHuggingFace wrapper
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    # Step 4: Create the LangChain Agent (using langgraph)
    agent = create_react_agent(llm, tools)
    
    return agent

async def run_travel_agent(user_request: str):
    """Run the travel agent with a user request."""
    response = await agent.ainvoke({"messages": [{"role": "user", "content": user_request}]})
    # Extract the last message content
    if response.get("messages"):
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            return last_message.content
    return str(response)

async def main():
    """Main function to test the agent."""
    await initialize_agent()
    
    # Test the agent
    print("=" * 50)
    print("Testing the Travel Agent")
    print("=" * 50)
    
    result = await run_travel_agent("Estimate the budget for a 5-day trip to Paris")
    print(f"\nAgent Response: {result}")

if __name__ == "__main__":
    asyncio.run(main())
