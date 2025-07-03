"""Detecting any Todo in repository and creating a GitHub issue for it."""

import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
# === CONFIGURATION ===
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")  
GITHUB_REPONAME = os.getenv("GITHUB_REPONAME")  
GITHUB_REPO = f"{GITHUB_USERNAME}/{GITHUB_REPONAME}"
REPO_PATH = Path(".")  # Use current directory or provide absolute path
INCLUDE_EXTENSIONS = {".py", ".txt", ".ts"}  # Adjust as needed

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") #you wll have to manually generate this locally
'''
Instructions:
Generate a github token for the repository, giving it read and write access to issues.
Store it in a .env file in the root of your project with the line:
GITHUB_TOKEN= <your_generated_token_here>
This will be exclusive to your forked repository.
.env files are already in .gitignore, so it won't be pushed to the remote repository.
'''

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
COMMENT_REGEX = re.compile(r"(#|//|--|<!--|/\*|\*)\s*TODO[:\s]+(.+)", re.IGNORECASE)

# === CORE LOGIC ===

def find_todos(file_path):
    todos = []
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                match = COMMENT_REGEX.search(line)
                if match:
                    todos.append((lineno, match.group(2).strip()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return todos

def get_existing_issues():
    issues = {}
    page = 1
    while True:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/issues?state=open&page={page}"
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for issue in batch:
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
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN must be set in your environment")

    existing_issues = get_existing_issues()

    for path in REPO_PATH.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue

        todos = find_todos(path)
        for lineno, comment in todos:
            marker = f"[AUTO-TODO] {path.relative_to(REPO_PATH)}:{lineno}"
            title = f"TODO in {path.name} line {lineno}"
            body = f"**Comment:**\n```\n{comment}\n```\n\n---\n{marker}"
            if not any(marker in issue_body for issue_body in existing_issues.values()):
                create_issue(title, body)

if __name__ == "__main__":
    main()
