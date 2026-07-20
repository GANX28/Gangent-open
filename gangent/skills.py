"""Gangent skill loading and context injection.

Gangent skills are runtime-side capability modules. They are not Codex skills.
Each skill is a small directory containing a manifest.json and skill.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import TaskInput


DEFAULT_SKILLS_DIR = "skills"


@dataclass(frozen=True)
class SkillManifest:
    """Machine-readable skill metadata."""

    name: str
    description: str
    when_to_use: list[str]
    recommended_tools: list[str]
    risk_notes: list[str]
    output_contract: str


@dataclass(frozen=True)
class Skill:
    """Loaded Gangent skill."""

    manifest: SkillManifest
    instructions: str
    path: Path


def default_skills_path(workspace_root: str) -> Path:
    """Return the default skills directory for a workspace."""

    return Path(workspace_root).resolve() / DEFAULT_SKILLS_DIR


def load_skills(path: str | Path) -> list[Skill]:
    """Load all valid skills from a skills directory."""

    root = Path(path)
    if not root.exists():
        return []
    skills: list[Skill] = []
    for skill_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        manifest_path = skill_dir / "manifest.json"
        instructions_path = skill_dir / "skill.md"
        if not manifest_path.exists() or not instructions_path.exists():
            continue
        manifest = _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        instructions = instructions_path.read_text(encoding="utf-8")
        skills.append(Skill(manifest=manifest, instructions=instructions, path=skill_dir))
    return skills


def resolve_skills(task_input: TaskInput, skills: list[Skill], limit: int = 2) -> list[Skill]:
    """Select relevant skills for a task using deterministic keyword scoring."""

    text = f"{task_input.goal}\n{task_input.user_message}".lower()
    scored: list[tuple[int, Skill]] = []
    for skill in skills:
        score = 0
        for phrase in skill.manifest.when_to_use:
            if phrase.lower() in text:
                score += 2
        for tool in skill.manifest.recommended_tools:
            if tool.lower() in text:
                score += 1
        if score > 0:
            scored.append((score, skill))
    scored.sort(key=lambda item: (-item[0], item[1].manifest.name))
    return [skill for _, skill in scored[: max(1, limit)]]


def skill_context(skills: list[Skill]) -> str:
    """Format selected skills into a prompt constraint block."""

    if not skills:
        return ""
    sections: list[str] = ["Selected Gangent skills:"]
    for skill in skills:
        sections.append(
            "\n".join(
                [
                    f"- name: {skill.manifest.name}",
                    f"  description: {skill.manifest.description}",
                    f"  recommended_tools: {', '.join(skill.manifest.recommended_tools)}",
                    f"  risk_notes: {'; '.join(skill.manifest.risk_notes)}",
                    f"  output_contract: {skill.manifest.output_contract}",
                    "  instructions:",
                    _indent(skill.instructions.strip(), "    "),
                ]
            )
        )
    return "\n".join(sections)


def inject_skill_context(task_input: TaskInput, skills: list[Skill]) -> TaskInput:
    """Return a TaskInput with selected skill context appended to constraints."""

    context = skill_context(skills)
    if not context:
        return task_input
    return TaskInput(
        goal=task_input.goal,
        user_message=task_input.user_message,
        workspace_root=task_input.workspace_root,
        constraints=[*task_input.constraints, context],
        created_at=task_input.created_at,
    )


def _manifest_from_dict(data: dict) -> SkillManifest:
    return SkillManifest(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        when_to_use=list(data.get("when_to_use", [])),
        recommended_tools=list(data.get("recommended_tools", [])),
        risk_notes=list(data.get("risk_notes", [])),
        output_contract=str(data.get("output_contract", "")),
    )


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
