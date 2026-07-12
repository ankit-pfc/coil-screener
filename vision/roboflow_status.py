from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


API_URL = "https://api.roboflow.com"


def project_parts(project_id: str) -> tuple[str | None, str]:
    parts = [part for part in project_id.strip("/").split("/") if part]
    if not parts:
        raise ValueError("Roboflow project id is required.")
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def model_parts(model_id: str) -> tuple[str | None, str, str]:
    parts = [part for part in model_id.strip("/").split("/") if part]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError("Roboflow model id must look like project/version or workspace/project/version.")


def _json_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text[:500]}


def _get(api_key: str, path: str, endpoint: str) -> dict[str, Any]:
    response = requests.get(
        f"{endpoint.rstrip('/')}/{path.strip('/')}",
        params={"api_key": api_key},
        timeout=45,
    )
    return {
        "status": response.status_code,
        "ok": response.ok,
        "data": _json_response(response),
    }


def _post(api_key: str, path: str, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{endpoint.rstrip('/')}/{path.strip('/')}",
        params={"api_key": api_key},
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=45,
    )
    return {
        "status": response.status_code,
        "ok": response.ok,
        "data": _json_response(response),
    }


def _project_summary(payload: dict[str, Any]) -> dict[str, Any]:
    project = payload.get("project", payload)
    return {
        "id": project.get("id"),
        "name": project.get("name"),
        "type": project.get("type"),
        "annotation": project.get("annotation"),
        "images": project.get("images"),
        "unannotated": project.get("unannotated"),
        "splits": project.get("splits"),
        "classes": project.get("classes"),
        "versions": project.get("versions"),
        "updated": project.get("updated"),
    }


def build_status_report(
    *,
    api_key: str,
    workspace: str,
    project: str,
    version: str | None = None,
    endpoint: str = API_URL,
) -> dict[str, Any]:
    project_response = _get(api_key, f"{workspace}/{project}", endpoint)
    batches_response = _get(api_key, f"{workspace}/{project}/batches", endpoint)
    search_response = _post(
        api_key,
        f"{workspace}/search/v1",
        endpoint,
        {
            "query": f"project:{project}",
            "pageSize": 25,
            "fields": ["filename", "tags", "split", "width", "height", "projectData"],
        },
    )
    version_response = None
    if version is not None:
        version_response = _get(api_key, f"{workspace}/{project}/{version}", endpoint)

    search_data = search_response.get("data") or {}
    search_results = search_data.get("results") or []
    examples = [
        {
            "id": result.get("id"),
            "filename": result.get("filename"),
            "tags": result.get("tags"),
            "projectData": result.get("projectData"),
        }
        for result in search_results[:10]
    ]
    batches_data = batches_response.get("data") or {}
    batches = [
        {
            "id": batch.get("id"),
            "name": batch.get("name"),
            "images": batch.get("images"),
            "numJobs": batch.get("numJobs"),
        }
        for batch in batches_data.get("batches", [])[:10]
    ]
    project_summary = _project_summary(project_response.get("data") or {})
    blockers: list[str] = []
    if int(project_summary.get("versions") or 0) == 0:
        blockers.append("No trained Roboflow versions exist for this project.")
    if int(project_summary.get("unannotated") or 0) > 0:
        blockers.append("Uploaded images are still unannotated.")
    if version_response and version_response.get("status") == 404:
        blockers.append(f"Requested Roboflow model version {project}/{version} is not available.")

    return {
        "workspace": workspace,
        "project": project,
        "version": version,
        "project_status": project_response.get("status"),
        "project": project_summary,
        "version_status": version_response.get("status") if version_response else None,
        "batches_status": batches_response.get("status"),
        "batches": batches,
        "search_status": search_response.get("status"),
        "search_total": search_data.get("total"),
        "uploaded_examples": examples,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print sanitized Roboflow project status.")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE"))
    parser.add_argument("--project-id", default=os.getenv("ROBOFLOW_PROJECT_ID") or os.getenv("ROBOFLOW_PROJECT"))
    parser.add_argument("--model-id", default=os.getenv("ROBOFLOW_MODEL_ID"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is required.")

    workspace = args.workspace
    project = None
    version = None
    if args.model_id:
        model_workspace, model_project, version = model_parts(args.model_id)
        workspace = workspace or model_workspace
        project = model_project
    if args.project_id:
        project_workspace, project_slug = project_parts(args.project_id)
        workspace = workspace or project_workspace
        project = project or project_slug
    if not workspace or not project:
        raise SystemExit("--workspace and --project-id are required unless --model-id includes workspace.")

    report = build_status_report(
        api_key=api_key,
        workspace=workspace,
        project=project,
        version=version,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
