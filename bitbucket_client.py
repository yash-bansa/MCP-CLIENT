import os
import json
import traceback
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

# === Load env variables ===
load_dotenv()

# === Logging Setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# === Configuration ===
ENDPOINT = os.getenv("AI_GATEWAY_ENDPOINT", "")
MODEL_NAME = os.getenv("AI_GATEWAY_MODEL_NAME", "")
SUBSCRIPTION_KEY = os.getenv("AI_GATEWAY_API_KEY", "")
SSE_URL = "http://localhost:8000/stream/sse"
MAX_TOKENS = 1000
TEMPERATURE = 0.1
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {SUBSCRIPTION_KEY}"
}

def _extract_json_block(text: str) -> str:
    """Extract JSON text from a markdown code block if present."""
    if not text:
        return ""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text.strip()

# === FastAPI App ===
app = FastAPI()

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: ChatQuery):
    logger.info(f"Received chat query: {query.query}")
    result = await main(query.query)
    logger.info(f"Returning result: {result}")
    return result

async def main(query: str):
    logger.info("Connecting to SSE tool server...")
    final_result = {}

    try:
        async with sse_client(url=SSE_URL, headers={}) as (in_stream, out_stream):
            async with ClientSession(in_stream, out_stream) as session:
                info = await session.initialize()
                logger.info(f"Connected to {info.serverInfo.name} v{info.serverInfo.version}")

                tools = await session.list_tools()
                available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools.tools]
                logger.info(f"Available tools: {[t['name'] for t in available_tools]}")

                messages = [
                    {
                        "role": "user",
                        "content": f"You are an assistant that can call external tools via JSON-formatted instructions. "
                                   f"Here are the available tools: {json.dumps(available_tools)}"
                    },
                    {
                        "role": "user",
                        "content": f"""When you need to use a tool, respond ONLY with a JSON object like:
{{"tool_name": "toolA", "arguments": {{"param1": "value1"}}}}
Otherwise, respond normally with plain text. The User Query is: {query}"""
                    }
                ]

                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS
                }

                # === Initial LLM call ===
                async with httpx.AsyncClient() as client:
                    http_response = await client.post(ENDPOINT, headers=HEADERS, json=payload)

                logger.debug(f"LLM Response Text: {http_response.text}")

                try:
                    result = http_response.json()
                    if "choices" not in result:
                        final_result["error"] = "Invalid response from LLM"
                        return final_result

                    reply_content = result['choices'][0]['message']['content']
                    logger.debug(f"Raw reply_content: {reply_content}")

                    # Clean markdown fences
                    cleaned_content = _extract_json_block(reply_content)
                    logger.debug(f"Cleaned JSON string: {cleaned_content}")

                except Exception:
                    logger.error("LLM response JSON error:\n" + traceback.format_exc())
                    final_result["error"] = "Malformed LLM response"
                    return final_result

                try:
                    tool_call = json.loads(cleaned_content)
                    tool_name = tool_call.get("tool_name")
                    tool_args = tool_call.get("arguments")

                    if tool_name:
                        logger.info(f"Calling tool '{tool_name}' with args: {tool_args}")
                        tool_result = await session.call_tool(tool_name, tool_args)

                        final_result.update({
                            "tool_name": tool_name,
                            "tool_arguments": tool_args,
                            "tool_response": tool_result.content
                        })

                        # Send tool result back to LLM for final message
                        messages.extend([
                            {"role": "assistant", "content": json.dumps(tool_call)},
                            {"role": "user", "content": f"Tool `{tool_name}` responded with: {tool_result.content}"}
                        ])
                        payload["messages"] = messages

                        async with httpx.AsyncClient() as client:
                            http_response = await client.post(ENDPOINT, headers=HEADERS, json=payload)

                        result = http_response.json()
                        reply_after_tool = result['choices'][0]['message']['content']
                        final_result["final_response"] = reply_after_tool
                    else:
                        # No tool call — just return cleaned content
                        final_result["final_response"] = cleaned_content

                except json.JSONDecodeError:
                    logger.warning("Reply content was not valid JSON, returning raw content.")
                    final_result["final_response"] = reply_content
                except Exception:
                    logger.error("Tool call error:\n" + traceback.format_exc())
                    final_result["final_response"] = reply_content

    except Exception:
        logger.error("Main flow error:\n" + traceback.format_exc())
        final_result["error"] = "Unhandled error in main"

    return final_result

# === For Local Testing ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)