from typing import cast

import chainlit as cl
from chainlit.types import AskFileResponse

from langchain_core.messages import HumanMessage, BaseMessageChunk
from langgraph.graph.graph import CompiledGraph

import agent1

def _agent():
    return cast(CompiledGraph, cl.user_session.get("agent"))

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="処理したい帳票をアップロードしてください。", accept=["application/pdf"]
        ).send()
    file = files[0]
    elements = [
      cl.Pdf(name=f"{file.name}", display="inline", path=f"{file.path}", page=1)
    ]
    await cl.Message(content=f"`{file.name}`がアップロードされました。", elements=elements).send()
    cl.user_session.set("file", file)

    agent = await agent1.initialize_agent()
    cl.user_session.set("agent", agent)

    actions = [
        cl.Action(
            name="action",
            icon="mouse-pointer-click",
            payload={"value": "配送依頼情報を抜き出してください。"},
            label="配送依頼情報の抽出"
        ),
        cl.Action(
            name="action",
            icon="mouse-pointer-click",
            payload={"value": "配送依頼情報を抜き出して、「配送依頼」アプリに入力してください。"},
            label="配送依頼情報の抽出と、Kintoneアプリへの入力"
        )
    ]
    await cl.Message(content="Interact with this action button:", actions=actions).send()

@cl.action_callback("action")
async def on_action(action: cl.Action):
    await stream(action.payload["value"])

@cl.on_message
async def main(message: cl.Message):
    await stream(message.content)

async def stream(content: str):
    answer = cl.Message(content="")
    await answer.send()

    file = cast(AskFileResponse, cl.user_session.get("file"))

    config = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    async for chunk in agent.astream({"messages": [message]}, config):
        for _, v in chunk.items():
            if "messages" in v:
                for m in v["messages"]:
                    if isinstance(m, BaseMessage):
                        print(m.pretty_repr())
                    else:
                        print(m)
            else:
                print(v)

    for msg, _ in _agent().stream(
        {"messages": [
            HumanMessage(content=f"{content} ファイルのパス:{file.path}")]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, BaseMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
