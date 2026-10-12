"""Тесты подключения к нескольким MCP-серверам (`MultiMCPClient`,
`parse_mcp_servers`) и передачи `_meta` с чатом и агентом."""

from __future__ import annotations

import unittest

from agents_core.mcp_client import (
    MCPClient, MCPClientError, MultiMCPClient, build_mcp_client, parse_mcp_servers,
)
from tests.test_mcp_client import FakeSession, _rpc_result


def _tools(*names, group=None):
    tools = [{"name": n, "title": f"Т {n}", "inputSchema": {"type": "object"}} for n in names]
    if group:
        for t in tools:
            t["_meta"] = {"agentscore/group": group}
    return _rpc_result({"tools": tools})


def _client(session) -> MCPClient:
    return MCPClient("http://x/mcp", session=session, max_retries=0)


class ParseServersTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(
            parse_mcp_servers("tools=http://a:8001/mcp, scheduler=http://b:8002/mcp"),
            [("tools", "http://a:8001/mcp"), ("scheduler", "http://b:8002/mcp")],
        )

    def test_bad_item_rejected(self):
        with self.assertRaises(ValueError):
            parse_mcp_servers("http://a:8001/mcp")

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            parse_mcp_servers("a=http://x,a=http://y")

    def test_single_server_is_plain_client(self):
        self.assertIsInstance(build_mcp_client([("tools", "http://a/mcp")]), MCPClient)
        self.assertIsInstance(build_mcp_client([("a", "http://a/mcp"), ("b", "http://b/mcp")]), MultiMCPClient)


class MultiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools_session = FakeSession([_tools("git_host_get_repo", "list_jobs")])
        self.sched_session = FakeSession([_tools("schedule_job", "list_jobs")])
        self.multi = MultiMCPClient({
            "tools": _client(self.tools_session), "scheduler": _client(self.sched_session),
        })

    def test_merges_and_prefixes_conflicting_names(self):
        names = {t["function"]["name"] for t in self.multi.list_tools()}
        self.assertEqual(names, {"git_host_get_repo", "schedule_job", "tools__list_jobs", "scheduler__list_jobs"})

    def test_details_carry_server_and_title(self):
        self.tools_session._responses.append(_tools("git_host_get_repo"))
        self.sched_session._responses.append(_tools("schedule_job"))
        self.multi.list_tools()
        details = {d["name"]: d for d in self.multi.list_tool_details()}
        self.assertEqual(details["schedule_job"]["server"], "scheduler")
        self.assertEqual(details["schedule_job"]["title"], "Т schedule_job")

    def test_group_from_tool_meta_else_server_name(self):
        self.tools_session._responses = [_tools("git_host_get_repo", group="GIT API")]
        self.sched_session._responses = [_tools("schedule_job")]
        details = {d["name"]: d for d in self.multi.list_tool_details()}
        self.assertEqual(details["git_host_get_repo"]["group"], "GIT API")
        # Сервер не сообщил группу — подпись по имени сервера из MCP_SERVERS.
        self.assertEqual(details["schedule_job"]["group"], "scheduler")
        self.assertEqual(self.multi.tool_group("git_host_get_repo"), "GIT API")
        self.assertIsNone(self.multi.tool_group("nope"))

    def test_single_server_keeps_configured_name(self):
        client = MCPClient("http://x/mcp", session=FakeSession([_tools("fetch")]), max_retries=0, name="ext")
        self.assertEqual(client.list_tool_details()[0]["group"], "ext")
        self.assertEqual(client.tool_group("fetch"), "ext")

    def test_call_routes_to_owner_with_original_name_and_meta(self):
        self.multi.list_tools()
        self.sched_session._responses.append(_rpc_result({"structuredContent": {"result": {"ok": True}}}))
        result = self.multi.call_tool("scheduler__list_jobs", {}, meta={"agentscore/chat_id": "c1"})
        self.assertEqual(result, {"ok": True})
        params = self.sched_session.calls[-1]["json"]["params"]
        self.assertEqual(params["name"], "list_jobs")
        self.assertEqual(params["_meta"], {"agentscore/chat_id": "c1"})
        self.assertTrue(self.multi.has_cached_tool("git_host_get_repo"))

    def test_allowed_tools_translated_to_server_names(self):
        self.multi.list_tools()
        self.sched_session._responses.append(_rpc_result({"structuredContent": {"result": {"ok": True}}}))
        self.multi.call_tool("scheduler__list_jobs", {}, meta={
            "agentscore/chat_id": "c1",
            "agentscore/allowed_tools": ["scheduler__list_jobs", "schedule_job", "tools__list_jobs", "git_host_get_repo"],
        })
        sent = self.sched_session.calls[-1]["json"]["params"]["_meta"]["agentscore/allowed_tools"]
        # Только инструменты этого сервера и без префикса.
        self.assertEqual(sent, ["list_jobs", "schedule_job"])
        self.assertFalse(self.multi.has_cached_tool("list_jobs"))

    def test_unknown_tool_raises(self):
        self.multi.list_tools()
        with self.assertRaises(MCPClientError):
            self.multi.call_tool("nope", {})


if __name__ == "__main__":
    unittest.main()
