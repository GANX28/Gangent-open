import tempfile
import unittest
from pathlib import Path

from gangent.memory_graph import (
    JsonMemoryGraphStore,
    MemoryEdgeType,
    MemoryLayer,
    MemoryNodeType,
    apply_decay,
    assemble_memory_context,
    build_memory_context_pack,
    default_memory_graph_path,
    default_memory_graph_viewer_data_path,
    memory_context_for_query,
    normalize_llm_memory_chunks,
    record_task_result_memory,
    retrieve_memory_graph,
    summarize_task_result,
)


class MemoryGraphTests(unittest.TestCase):
    def test_store_saves_and_loads_nodes_and_edges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json"
            store = JsonMemoryGraphStore(path)
            issue = store.add_node(
                MemoryNodeType.ISSUE,
                "DeepSeek returned plain text tool request.",
                project_scope="gangent",
                source="tests",
            )
            fix = store.add_node(MemoryNodeType.SOLUTION, "Reject fake tool request direct responses.")
            store.add_edge(issue.node_id, fix.node_id, MemoryEdgeType.FIXED_BY)
            store.save()

            loaded = JsonMemoryGraphStore(path)

            self.assertEqual(len(loaded.nodes), 2)
            self.assertEqual(len(loaded.edges), 1)
            self.assertEqual(loaded.edges[0].edge_type, MemoryEdgeType.FIXED_BY)
            self.assertTrue((Path(temp_dir) / "graph_data.js").exists())

    def test_default_save_exports_viewer_data_js(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = default_memory_graph_path(temp_dir)
            store = JsonMemoryGraphStore(path)
            store.add_node(MemoryNodeType.CONCEPT, "Dynamic context uses task-specific memory packs.")
            store.save()

            viewer_data = default_memory_graph_viewer_data_path(temp_dir)

            self.assertTrue(viewer_data.exists())
            self.assertIn("window.GANGENT_MEMORY_GRAPH", viewer_data.read_text(encoding="utf-8"))
            self.assertIn("Dynamic context", viewer_data.read_text(encoding="utf-8"))

    def test_retrieve_expands_graph_neighbors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json"
            store = JsonMemoryGraphStore(path)
            issue = store.add_node(MemoryNodeType.ISSUE, "JSON drift in document extraction")
            solution = store.add_node(MemoryNodeType.SOLUTION, "Validate JSON schema and retry repair")
            store.add_edge(issue.node_id, solution.node_id, MemoryEdgeType.FIXED_BY, weight=1.0)

            results = retrieve_memory_graph("document JSON drift", store, top_k=5, max_depth=1)
            ids = {result.node.node_id for result in results}

            self.assertIn(issue.node_id, ids)
            self.assertIn(solution.node_id, ids)
            self.assertGreater(store.nodes[issue.node_id].access_count, 0)

    def test_assemble_memory_context_formats_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonMemoryGraphStore(Path(temp_dir) / "graph.json")
            node = store.add_node(
                MemoryNodeType.CONSTRAINT,
                "Do not claim production ownership for a prototype.",
                project_scope="resume",
                source="handoff",
            )
            results = retrieve_memory_graph("prototype production ownership", store)

            context = assemble_memory_context(results)

            self.assertIn("Relevant Memory", context)
            self.assertIn(node.node_type.value, context)
            self.assertIn("production ownership", context)

    def test_apply_decay_marks_stale_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonMemoryGraphStore(Path(temp_dir) / "graph.json")
            node = store.add_node(MemoryNodeType.NOTE, "old note")
            node.decay_score = 0.2

            changed = apply_decay(store, decay_factor=0.5, stale_threshold=0.15)

            self.assertEqual(changed, 1)
            self.assertTrue(node.stale)

    def test_memory_context_for_query_uses_default_workspace_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = default_memory_graph_path(temp_dir)
            store = JsonMemoryGraphStore(path)
            store.add_node(MemoryNodeType.DECISION, "Use repo-root mode for self-hosting Gangent.")
            store.save()

            context = memory_context_for_query("repo-root Gangent", temp_dir)

            self.assertIn("repo-root mode", context)

    def test_memory_context_pack_groups_data_task_and_knowledge_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonMemoryGraphStore(Path(temp_dir) / "graph.json")
            data = store.add_node(MemoryNodeType.ARTIFACT, "README content", layer=MemoryLayer.DATA)
            task = store.add_node(MemoryNodeType.DECISION, "Use bounded planner steps")
            knowledge = store.add_node(MemoryNodeType.CONCEPT, "Correct context beats larger context")

            results = retrieve_memory_graph("context planner README", store, top_k=5)
            pack = build_memory_context_pack(results)
            context = assemble_memory_context(results)

            self.assertIn(data.node_id, {item.node.node_id for item in pack.data_nodes})
            self.assertIn(task.node_id, {item.node.node_id for item in pack.task_nodes})
            self.assertIn(knowledge.node_id, {item.node.node_id for item in pack.knowledge_nodes})
            self.assertIn("Knowledge layer", context)

    def test_record_task_result_memory_writes_log_and_semantic_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = record_task_result_memory(
                workspace_root=temp_dir,
                task_id="task_test",
                session_id="session_test",
                user_message="Read README and summarize Gangent.",
                status="completed",
                final_answer="Gangent is a local agent runtime for dynamic context and tool control.",
                errors=["Internal warning kept for maintenance."],
                tool_names=["read_file", "finish_task"],
            )

            store = JsonMemoryGraphStore(path)
            viewer_data = default_memory_graph_viewer_data_path(temp_dir)
            task_log = default_memory_graph_path(temp_dir).with_name("task_log.jsonl")

            self.assertTrue(path.exists())
            self.assertTrue(viewer_data.exists())
            self.assertTrue(task_log.exists())
            self.assertFalse(any("task_result" in node.tags for node in store.nodes.values()))
            self.assertTrue(any("semantic_chunk" in node.tags for node in store.nodes.values()))
            self.assertFalse(any(node.node_type == MemoryNodeType.ISSUE for node in store.nodes.values()))
            self.assertIn("window.GANGENT_MEMORY_GRAPH", viewer_data.read_text(encoding="utf-8"))

    def test_reusable_runtime_issue_is_linked_to_semantic_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = record_task_result_memory(
                workspace_root=temp_dir,
                task_id="task_test",
                session_id="session_test",
                user_message="Analyze planner stability.",
                status="completed",
                final_answer="Planner stability depends on phase-aware replanning and bounded recovery.",
                errors=["Plan guard: tool call is outside the current plan phase."],
                tool_names=["read_file"],
            )

            store = JsonMemoryGraphStore(path)
            issues = [node for node in store.nodes.values() if node.node_type == MemoryNodeType.ISSUE]
            linked_ids = {
                edge.source_node_id
                for edge in store.edges
            } | {
                edge.target_node_id
                for edge in store.edges
            }

            self.assertEqual(len(issues), 1)
            self.assertIn(issues[0].node_id, linked_ids)

    def test_record_task_result_memory_uses_final_answer_when_request_is_corrupt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = record_task_result_memory(
                workspace_root=temp_dir,
                task_id="task_test",
                session_id="session_test",
                user_message="????????????????",
                status="completed",
                final_answer="Planner analysis completed with deterministic plan generation details.",
            )

            store = JsonMemoryGraphStore(path)
            semantic_nodes = [
                node
                for node in store.nodes.values()
                if "semantic_chunk" in node.tags and "task_memory" in node.tags
            ]

            self.assertEqual(len(semantic_nodes), 1)
            self.assertIn("Planner analysis completed", semantic_nodes[0].summary)
            self.assertNotIn("????", semantic_nodes[0].summary)

    def test_summarize_task_result_skips_non_reusable_chat(self):
        chunks = summarize_task_result(
            task_id="task_test",
            user_message="Say hello.",
            status="completed",
            final_answer="Hello.",
            tool_names=[],
        )

        self.assertEqual(chunks, [])

    def test_record_task_result_memory_prefers_valid_llm_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = record_task_result_memory(
                workspace_root=temp_dir,
                task_id="task_test",
                session_id="session_test",
                user_message="Analyze memory graph design.",
                status="completed",
                final_answer="Memory graph uses semantic chunks.",
                llm_chunks=[
                    {
                        "node_type": "concept",
                        "layer": "knowledge",
                        "summary": "Memory graph nodes are semantic chunks for dynamic context routing.",
                        "content": "Each node stores a short routing summary and bounded detail. Retrieval selects relevant nodes and expands through graph edges before building a context pack.",
                        "tags": ["memory", "context"],
                        "importance": 0.9,
                        "confidence": 0.85,
                    }
                ],
            )

            store = JsonMemoryGraphStore(path)
            nodes = list(store.nodes.values())

            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].node_type, MemoryNodeType.CONCEPT)
            self.assertIn("llm_extracted", nodes[0].tags)
            self.assertIn("dynamic context routing", nodes[0].summary)

    def test_normalize_llm_memory_chunks_rejects_invalid_payload(self):
        chunks = normalize_llm_memory_chunks(
            [
                {"node_type": "bad", "summary": "Valid looking summary", "content": "Long enough content"},
                {"node_type": "concept", "summary": "????", "content": "????????????"},
                {
                    "node_type": "decision",
                    "layer": "task",
                    "summary": "Use bounded context packs for memory routing.",
                    "content": "The runtime should load selected memory node details only after routing by summary and edges.",
                    "tags": ["Context Pack"],
                },
            ],
            task_id="task_test",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["node_type"], MemoryNodeType.DECISION)
        self.assertIn("context-pack", chunks[0]["tags"])

    def test_retrieval_ignores_legacy_task_result_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonMemoryGraphStore(Path(temp_dir) / "graph.json")
            legacy = store.add_node(
                MemoryNodeType.TASK_STATE,
                "Task ID: task_old\nFinal answer: context pollution detail",
                summary="completed: context pollution detail",
                tags=["task_result", "completed"],
                layer=MemoryLayer.TASK,
            )
            concept = store.add_node(
                MemoryNodeType.CONCEPT,
                "Context pollution should be reduced with task-specific memory chunks.",
                summary="Use semantic memory chunks to reduce context pollution.",
                tags=["semantic_chunk"],
                layer=MemoryLayer.KNOWLEDGE,
            )

            results = retrieve_memory_graph("context pollution", store, top_k=5)
            ids = {result.node.node_id for result in results}

            self.assertNotIn(legacy.node_id, ids)
            self.assertIn(concept.node_id, ids)


if __name__ == "__main__":
    unittest.main()
