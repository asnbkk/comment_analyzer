"""Fetch YouTube comments using the official YouTube Data API v3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


ENV_PATH = Path(".env")
VIDEO_ID = "t_HtRmlKCVI"
ORDER = "relevance"  # "time" or "relevance"
OUTPUT_PATH = Path("comments_youtube_api.jsonl")


def load_env_file(path: Path) -> None:
    """Populate os.environ with key/value pairs from a .env file if it exists."""

    if not path.is_file():
        return

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def get_api_key() -> str:
    """Read the YouTube API key from environment variables."""

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable YOUTUBE_API_KEY is not set. "
            "Store your key in .env and export it before running."
        )
    return api_key


def build_client(api_key: str):
    """Create a YouTube API client using the provided API key."""

    return build("youtube", "v3", developerKey=api_key)


def fetch_comment_threads(
    youtube,
    video_id: str,
    order: str = "time",
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch top-level comment threads for the video."""

    threads: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    pages_fetched = 0

    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            order=order,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
        )
        response = request.execute()
        threads.extend(response.get("items", []))

        pages_fetched += 1
        if max_pages is not None and pages_fetched >= max_pages:
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return threads


def fetch_replies_for_thread(
    youtube,
    thread: Dict[str, Any],
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch all replies for a given top-level comment thread."""

    snippet = thread.get("snippet", {})
    top_level = snippet.get("topLevelComment", {})
    parent_id = top_level.get("id")
    total_reply_count = snippet.get("totalReplyCount", 0)

    initial_replies = thread.get("replies", {}).get("comments", [])
    if total_reply_count <= len(initial_replies) or not parent_id:
        return initial_replies

    replies: List[Dict[str, Any]] = list(initial_replies)
    page_token: Optional[str] = None
    pages_fetched = 0

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText",
        )
        response = request.execute()
        replies.extend(response.get("items", []))

        pages_fetched += 1
        if max_pages is not None and pages_fetched >= max_pages:
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return replies


def flatten_threads(
    threads: Iterable[Dict[str, Any]],
    replies_map: Dict[str, List[Dict[str, Any]]],
) -> Iterator[Dict[str, Any]]:
    """Yield top-level comments followed by their replies."""

    for thread in threads:
        top_level = thread.get("snippet", {}).get("topLevelComment")
        if top_level:
            yield top_level

        parent_id = top_level.get("id") if top_level else None
        if parent_id and parent_id in replies_map:
            yield from replies_map[parent_id]


def export_comments_to_jsonl(comments: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Write comments to a JSONL file."""

    with output_path.open("w", encoding="utf-8") as fh:
        for comment in comments:
            fh.write(json.dumps(comment, ensure_ascii=False) + "\n")


def main() -> None:
    load_env_file(ENV_PATH)
    api_key = get_api_key()
    youtube = build_client(api_key)

    try:
        threads = fetch_comment_threads(youtube, VIDEO_ID, order=ORDER)
    except HttpError as exc:
        raise SystemExit(f"YouTube API error: {exc}") from exc

    replies_by_parent: Dict[str, List[Dict[str, Any]]] = {}

    for thread in threads:
        top_level = thread.get("snippet", {}).get("topLevelComment")
        parent_id = top_level.get("id") if top_level else None
        if not parent_id:
            continue

        replies = fetch_replies_for_thread(youtube, thread)
        if replies:
            replies_by_parent[parent_id] = replies

    flattened_comments = list(flatten_threads(threads, replies_by_parent))
    export_comments_to_jsonl(flattened_comments, OUTPUT_PATH)

    total_comments = len(flattened_comments)
    print(f"Saved {total_comments} comment objects (including replies) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

