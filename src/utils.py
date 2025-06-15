import copy
from langchain_core.messages.base import BaseMessage

def fix_tool_schema(tools): 
    """kintone-mcp-serverがlangchain-googleが理解できない形式を返すので、動的に修正します。

    Args:
        tools (list): StructuredToolのリスト
    """
    def fix_schema(schema):
        # schema全体を再帰的に走査
        if isinstance(schema, dict):
            new_schema = {}
            for key, value in schema.items():
                if key == "type" and value == "number":
                    new_schema[key] = "integer"
                elif isinstance(value, (dict, list)):
                    new_schema[key] = fix_schema(value)
                else:
                    new_schema[key] = value
            return new_schema
        elif isinstance(schema, list):
            return [fix_schema(item) for item in schema]
        else:
            return schema

    fixed_tools = []

    for tool in tools:
        fixed_tool = copy.copy(tool)  # StructuredTool の shallow copy
        fixed_schema = fix_schema(tool.args_schema)
        fixed_tool.args_schema = fixed_schema
        fixed_tools.append(fixed_tool)

    return fixed_tools

def print_chunk(chunk):
    """Print the chunk."""
    for _, v in chunk.items():
        if "messages" in v:
            for m in v["messages"]:
                if isinstance(m, BaseMessage):
                    print(m.pretty_repr())
                else:
                    print(m)
        else:
            print(v)
