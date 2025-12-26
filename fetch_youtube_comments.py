"""Fetch YouTube comment pages using scrapecreators API."""

import json
import time
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "https://api.scrapecreators.com/v1/youtube/video/comments"
API_KEY = "H8O9EXtOlceiWYhOExkCyJvMP0z1"
VIDEO_URL = "https://www.youtube.com/watch?v=t_HtRmlKCVI"
ORDER = "top"


def fetch_comment_pages(
    video_url: str,
    api_key: str,
    order: Optional[str] = None,
    start_token: Optional[str] = None,
    max_pages: Optional[int] = None,
    sleep_seconds: float = 0.0,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    session = requests.Session()
    headers = {"x-api-key": api_key}
    pages: List[Dict[str, Any]] = []
    token = start_token
    page_count = 0

    while True:
        params = {"url": video_url}
        if order:
            params["order"] = order
        if token:
            params["continuationToken"] = token

        response = session.get(
            BASE_URL, headers=headers, params=params, timeout=timeout
        )
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        pages.append(data)

        page_count += 1
        if max_pages is not None and page_count >= max_pages:
            break

        token = data.get("continuationToken")
        if not token:
            break

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return pages


def main() -> None:
    pages = fetch_comment_pages(
        video_url=VIDEO_URL,
        api_key=API_KEY,
        order=ORDER,
        start_token=None,
        max_pages=None,
        sleep_seconds=0.2,
    )

    print(f"Fetched {len(pages)} page(s) of comments")

    all_comments: List[Dict[str, Any]] = []

    for idx, page in enumerate(pages, start=1):
        continuation = page.get("continuationToken")
        comments = page.get("comments")
        comment_count = len(comments) if isinstance(comments, list) else 0
        print(f"Page {idx}: {comment_count} comment(s)")
        if continuation:
            print(f"  Next continuationToken: {continuation[:60]}…")

        if isinstance(comments, list):
            all_comments.extend(comments)

    suffix = ORDER if ORDER else "default"
    output_path = f"/Users/assan/Desktop/coding/bkn_sn_analysis/comments_{suffix}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fh:
        for comment in all_comments:
            fh.write(json.dumps(comment, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_comments)} comment(s) to {output_path}")


if __name__ == "__main__":
    main()
