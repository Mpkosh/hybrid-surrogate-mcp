from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
# llm
import os
from openai import OpenAI
import httpx
#from azure.ai.inference import ChatCompletionsClient
#from azure.ai.inference.models import SystemMessage, UserMessage
#from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, find_dotenv
import json

# to override existing env.variables        
load_dotenv(find_dotenv(usecwd=True), override=True)

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="mcp",  # Executable
    args=["run", "server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

def convert_to_llm_tool(tool):
    tool_schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"]
            }
        }
    }

    return tool_schema


def call_llm(prompt, functions):
    #token = os.environ["GITHUB_TOKEN"]
    #endpoint = "https://models.inference.ai.azure.com"

    model_name = os.getenv("MODEL_NAME")
    httpx_client = httpx.Client(verify=False,trust_env=False,timeout=600)
    client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=7200,
        http_client=httpx_client
    )
    print(os.getenv("BASE_URL"),os.getenv("MODEL_NAME"))
    print("CALLING LLM")
    response = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": "You are a helpful assistant.",
            },
            {
            "role": "user",
            "content": prompt,
            },
        ],
        model=model_name,
        tools = functions,
        # Optional parameters
        temperature=1.,
        max_tokens=1000,
        extra_body={"reasoning_effort": "none"},
        top_p=1.    
    )

    response_message = response.choices[0].message#.content
    
    functions_to_call = []

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            print("TOOL: ", tool_call)
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            functions_to_call.append({ "name": name, "args": args })

    return functions_to_call


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            # Initialize the connection
            await session.initialize()

            # List available resources
            resources = await session.list_resources()
            print("LISTING RESOURCES")
            for resource in resources:
                print("Resource: ", resource)

            # List available tools
            tools = await session.list_tools()
            print("LISTING TOOLS")
            for tool in tools.tools:
                print("Tool: ", tool.name)

            # Read a resource
            print("READING RESOURCE")
            content, mime_type = await session.read_resource("greeting://hello")

            # Call a tool
            print("CALL TOOL")
            result = await session.call_tool("add", arguments={"a": 1, "b": 7})
            print(result.content)

            functions = []
            for tool in tools.tools:
                print("Tool: ", tool.name)
                print("Tool", tool.inputSchema["properties"])
                functions.append(convert_to_llm_tool(tool))

            prompts = ["Add 2 to 20", 'What is 6 divided by 2?', 
                       'I need to give 2 pens to each friend; I have 3 friends. How much pens should I buy?']
            for prompt in prompts:
                # ask LLM what tools to all, if any
                functions_to_call = call_llm(prompt, functions)

                # call suggested functions
                for f in functions_to_call:
                    result = await session.call_tool(f["name"], arguments=f["args"])
                    print("TOOLS result: ", result.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())