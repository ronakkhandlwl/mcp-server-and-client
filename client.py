from mcp import ClientSession, StdioServerParameters
from langchain_openai import ChatOpenAI
from mcp.client.stdio import stdio_client
import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

server_params = StdioServerParameters(
  command="python",
  args=["math_server.py"],
)

model = ChatOpenAI(model="gpt-4o", api_key=api_key)


async def run_agent():
  async with stdio_client(server_params) as (read, write):
    # Open an MCP session to interact with the math_server.py tool.
    async with ClientSession(read, write) as session:
      # Initialize the session.
      await session.initialize()
      # Load tools
      tools = await load_mcp_tools(session)
      # Create a ReAct agent.
      agent = create_react_agent(model, tools)
      # Run the agent.
      agent_response = await agent.ainvoke(
        # Now, let's give our message.
       {"input": "what's (4 + 6) x 14?"})
      # Return the response.
      return agent_response["output"]

if __name__ == "__main__":
  result = asyncio.run(run_agent())
  print(result)
