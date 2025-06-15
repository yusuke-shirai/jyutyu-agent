import asyncio
from doctest import debug
from dotenv import load_dotenv
import base64

from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

import utils
from utils import print_chunk

load_dotenv(override=True)


@tool(parse_docstring=True)
def extract_business_info_from_local_file(path: str) -> str:
    """Extract business information as JSON from the local file.

    Args:
        path (str): path for the local file

    Returns:
        str: JSON string with the business information
    """
    data = None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return _extract_business_info(
        {
            "type": "file",
            "source_type": "base64",
            "data": data,
            "mime_type": "application/pdf",
        }
    )


@tool(parse_docstring=True)
def extract_business_info_from_url(url: str) -> str:
    """Extract business information as JSON from the file at the URL.

    Args:
        url (str): URL for the file

    Returns:
        str: JSON string with the business information
    """
    return _extract_business_info(
        {
            "type": "file",
            "source_type": "url",
            "url": url,
        }
    )


def _extract_business_info(payload: dict) -> str:
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Read the given document and extract the business information as JSON. DON'T include any text other than JSON:",
            },
            payload,
        ],
    }
    llm = init_chat_model("google_genai:gemini-2.0-flash")
    response = llm.invoke([message])
    return response.text()


async def initialize_agent():
    client = MultiServerMCPClient(
        {
            "kintone": {
                "transport": "stdio",
                "command": "node",
                "args": ["./kintone-mcp-server/server.js"],
            }
        }
    )
    tools = await client.get_tools()
    # そのままGeminiに読ませるとエラーになるので、スキーマ定義を修正。
    tools = utils.fix_tool_schema(tools)
    # 上記をやってもエラーになってしまうのでスキップするツール名
    skip_tool_name_for_gemini = ("update_form_layout", "create_reference_table_field")
    tools = [t for t in tools if t.name not in skip_tool_name_for_gemini]
    tools.append(extract_business_info_from_url)
    tools.append(extract_business_info_from_local_file)

    agent = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        checkpointer=InMemorySaver(),
        tools=tools,
        prompt="""あなたはユーザーの指示に従い帳票処理を行う有能なエージェントです。
与えられたツールを適切に使いユーザーの期待に応えてください。
どのツールを何回繰り返し使うかも自分で考えてください。

【注意事項】
1) システムに入力するときは、まず、既存のレコードを取得し
・どのようなフィールドがあるか
・各フィールドの入力フォーマットはどのようなものか
を確認してください。その結果を踏まえ、抜き出した配送依頼情報からレコード登録用のリクエストを発行してください。
""",
    )
    return agent


async def main():
    agent = await initialize_agent()

    while True:
        instruction = input("指示をしてください:")
        if not instruction:
            instruction = "配送依頼情報を'./data/配送依頼書.pdf'から抜き出して、「配送依頼」アプリに入力してください。"
        if "quit" in instruction:
            break
        message = {"role": "user", "content": instruction}
        async for chunk in agent.astream({"messages": [message]}, debug=False):
            print_chunk(chunk)


if __name__ == "__main__":
    asyncio.run(main())
