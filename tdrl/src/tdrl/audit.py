import ast
import re
import tomllib
from pathlib import Path

FORBIDDEN = {
    "openai",
    "anthropic",
    "litellm",
    "cohere",
    "boto3",
    "google.generativeai",
    "google.genai",
    "azure.ai",
}


def static_audit(root: Path):
    violations = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            for name in imports:
                if any(name == p or name.startswith(p + ".") for p in FORBIDDEN):
                    violations.append(f"{path}:{node.lineno}: forbidden import {name}")
        for pattern in (
            r"https?://(?:api\.|[^/\s]*openai\.com)",
            r"(?:OPENAI|ANTHROPIC|AWS|AZURE)_\w*(?:KEY|TOKEN|SECRET)",
            r"TDRL_OR|emnlp26_crs",
        ):
            if path.name != "audit.py" and re.search(pattern, text):
                violations.append(f"{path}: forbidden inference/legacy/credential path")
    lock = root / "uv.lock"
    if lock.exists():
        for package in tomllib.loads(lock.read_text())["package"]:
            if package["name"] in FORBIDDEN:
                violations.append(f"{lock}: forbidden dependency {package['name']}")
    return violations


def dependency_audit(path: Path, allow_local_vllm=False):
    lock = tomllib.loads(path.read_text())
    approved = {"openai", "anthropic"} if allow_local_vllm else set()
    if approved:
        vllm = next((p for p in lock["package"] if p["name"] == "vllm"), None)
        root = next((p for p in lock["package"] if p["name"] == "tdrl-h100-runtime"), None)
        if not vllm or vllm["version"] not in {"0.30.0", "0.30.0+cu129"} or not root:
            raise ValueError("SDK exception requires the pinned local vLLM runtime project")
        if not approved.issubset({p["name"] for p in vllm["dependencies"]}):
            raise ValueError("SDK exception is limited to mandatory vLLM dependencies")
        if approved & {p["name"] for p in root["dependencies"]}:
            raise ValueError("SDKs cannot be direct application dependencies")
    forbidden = sorted(p["name"] for p in lock["package"] if p["name"] in FORBIDDEN - approved)
    return {
        "status": "blocked" if forbidden else "passed",
        "forbidden_packages": forbidden,
        "approved_local_server_only": sorted(approved),
        "source": str(path),
    }
