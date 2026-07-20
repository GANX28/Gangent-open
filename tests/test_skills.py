import json
import tempfile
import unittest
from pathlib import Path

from gangent.models import TaskInput
from gangent.skills import inject_skill_context, load_skills, resolve_skills


class SkillsTests(unittest.TestCase):
    def test_load_and_resolve_skill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = root / "runtime-architect"
            skill_dir.mkdir()
            (skill_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "runtime-architect",
                        "description": "Runtime architecture.",
                        "when_to_use": ["runtime"],
                        "recommended_tools": ["search_context"],
                        "risk_notes": ["Keep boundaries clear."],
                        "output_contract": "Explain structure.",
                    }
                ),
                encoding="utf-8",
            )
            (skill_dir / "skill.md").write_text("# Runtime Architect\nUse structure.", encoding="utf-8")
            task_input = TaskInput(
                goal="Improve runtime architecture",
                user_message="Improve runtime architecture",
                workspace_root=temp_dir,
            )

            skills = load_skills(root)
            selected = resolve_skills(task_input, skills)

            self.assertEqual(len(skills), 1)
            self.assertEqual(selected[0].manifest.name, "runtime-architect")

    def test_inject_skill_context_appends_constraints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = root / "safe-code-editor"
            skill_dir.mkdir()
            (skill_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "safe-code-editor",
                        "description": "Safe code edits.",
                        "when_to_use": ["code"],
                        "recommended_tools": ["apply_patch"],
                        "risk_notes": ["Inspect before editing."],
                        "output_contract": "Report verification.",
                    }
                ),
                encoding="utf-8",
            )
            (skill_dir / "skill.md").write_text("Use patch edits.", encoding="utf-8")
            task_input = TaskInput(goal="edit code", user_message="edit code", workspace_root=temp_dir)

            updated = inject_skill_context(task_input, resolve_skills(task_input, load_skills(root)))

            self.assertGreater(len(updated.constraints), len(task_input.constraints))
            self.assertIn("safe-code-editor", updated.constraints[-1])


if __name__ == "__main__":
    unittest.main()
