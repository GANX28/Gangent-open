import unittest

from gangent.mcp_adapter import MCPAdapterError, MCPToolSpec, adapt_mcp_tool, mcp_tool_id
from gangent.tool_registry import ToolRisk, ToolSource


class MCPAdapterTests(unittest.TestCase):
    def test_mcp_tool_id_uses_namespace(self):
        self.assertEqual(mcp_tool_id("filesystem", "read_file"), "mcp.filesystem.read_file")

    def test_adapt_mcp_tool_creates_tool_definition(self):
        definition = adapt_mcp_tool(
            MCPToolSpec(
                server_name="filesystem",
                tool_name="read_file",
                description="Read file through MCP.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                risk=ToolRisk.READ,
            )
        )

        self.assertEqual(definition.name, "mcp.filesystem.read_file")
        self.assertEqual(definition.source, ToolSource.MCP)
        self.assertEqual(definition.risk, ToolRisk.READ)
        self.assertEqual(definition.input_schema["name"], "mcp.filesystem.read_file")

    def test_mcp_proxy_is_explicitly_not_connected(self):
        definition = adapt_mcp_tool(
            MCPToolSpec(
                server_name="demo",
                tool_name="echo",
                description="Echo.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        )

        with self.assertRaises(MCPAdapterError):
            definition.handler({"text": "hello"}, ".")

    def test_rejects_loose_schema(self):
        with self.assertRaises(MCPAdapterError):
            adapt_mcp_tool(
                MCPToolSpec(
                    server_name="demo",
                    tool_name="echo",
                    description="Echo.",
                    input_schema={"type": "object", "properties": {}},
                )
            )


if __name__ == "__main__":
    unittest.main()
