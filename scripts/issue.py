import os
import re
import requests
from pathlib import Path

# === ENVIRONMENT VARIABLES FROM GITHUB ACTIONS ===
GITHUB_TOKEN = os.environ.get("GH_API_TOKEN")
GITHUB_REPO = f"{os.environ.get('GH_USERNAME')}/{os.environ.get('GH_REPO_NAME')}"

# === STATIC CONFIG ===
REPO_PATH = Path(__file__).resolve().parent.parent  # Root of the repo
INCLUDE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go"}
COMMENT_REGEX = re.compile(r"(#|//|--|<!--|/\*|\*)\s*TODO[:\s]+(.+)", re.IGNORECASE)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# === FUNCTIONS ===

def find_todos(file_path):
    todos = []
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                match = COMMENT_REGEX.search(line)
                if match:
                    todos.append((lineno, match.group(2).strip()))
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
    return todos

def get_existing_issues():
    issues = {}
    page = 1
    while True:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/issues?state=open&page={page}"
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        for issue in data:
            issues[issue["title"]] = issue["body"]
        page += 1
    return issues

def create_issue(title, body):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    resp = requests.post(url, headers=HEADERS, json={"title": title, "body": body})
    if resp.status_code == 201:
        print(f"✅ Created issue: {title}")
    else:
        print(f"❌ Failed to create issue: {title} — {resp.status_code}: {resp.text}")

def main():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise EnvironmentError("GH_API_TOKEN, GH_USERNAME, or GH_REPO_NAME missing from environment")

    existing_issues = get_existing_issues()

    for path in REPO_PATH.rglob("*"):
        if path.is_file() and path.suffix.lower() in INCLUDE_EXTENSIONS:
            todos = find_todos(path)
            for lineno, comment in todos:
                rel_path = path.relative_to(REPO_PATH)
                marker = f"[AUTO-TODO] {rel_path}:{lineno}"
                title = f"TODO in {rel_path.name} line {lineno}"
                body = f"**Comment:**\n```\n{comment}\n```\n\n---\n{marker}"
                if not any(marker in b for b in existing_issues.values()):
                    create_issue(title, body)

if __name__ == "__main__":
    main()
