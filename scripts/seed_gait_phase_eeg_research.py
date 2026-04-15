from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.control_plane.paths import get_control_plane_paths
from bci_autoresearch.control_plane.runtime_store import append_jsonl, read_jsonl


QUERY_SPECS = [
    {
        "query": "gait EEG stance swing timing window length temporal attention",
        "intent": "找步态脑电二分类里窗长和时间注意力怎么配合，优先看稳定区里模型该关注哪一段。",
    },
    {
        "query": "gait EEG premovement timing decoding fixed lag negative lag",
        "intent": "找脑电提前于动作以及轻微反馈时延的证据，帮助这轮 signed lag 扫描。",
    },
    {
        "query": "EEG gait phase decoding support swing attention pooling",
        "intent": "找支撑/摆动任务里 attention pooling 的先例，判断 masked safe band 有没有现实依据。",
    },
    {
        "query": "fixed-lag decoding gait EEG support swing classification window size",
        "intent": "找 window × lag 联合搜索的直接先例，帮助判断二维 timing scan 是否合理。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--topic-id", default="gait_phase_eeg_classification")
    parser.add_argument("--max-results-per-query", type=int, default=2)
    return parser.parse_args()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_europe_pmc(query: str, page_size: int) -> list[dict[str, Any]]:
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
        f"query={quote_plus(query)}&format=json&pageSize={page_size}"
    )
    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result_list = payload.get("resultList") or {}
    results = result_list.get("result") or []
    return [item for item in results if isinstance(item, dict)]


def build_source_url(item: dict[str, Any]) -> str:
    doi = str(item.get("doi") or "").strip()
    pmid = str(item.get("pmid") or "").strip()
    pmcid = str(item.get("pmcid") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return str(item.get("sourceUrl") or item.get("journalInfo") or "").strip()


def build_why_it_matters(query: str, title: str) -> str:
    lowered = f"{query} {title}".lower()
    if "premovement" in lowered or "movement-related" in lowered or "fixed-lag" in lowered:
        return "这条证据可以帮助判断脑电更像提前于动作，还是同步于动作最有信息。"
    if "window" in lowered or "size" in lowered or "length" in lowered:
        return "这条证据可以帮助判断 0.5s 到 3.0s 这些窗口里，哪些更可能装下足够的步态脑电信息。"
    if "phase" in lowered or "stance" in lowered or "swing" in lowered:
        return "这条证据和支撑/摆动二分类最接近，适合作为 timing scan 的直接任务参考。"
    return "这条证据可作为步态脑电 timing scan 的窗口、时延或任务定义参考。"


def main() -> None:
    args = parse_args()
    paths = get_control_plane_paths(ROOT)
    existing_rows = [row for row in read_jsonl(paths.research_evidence) if isinstance(row, dict)]
    seen_pairs = {
        (
            str(row.get("source_title") or "").strip(),
            str(row.get("source_url") or "").strip(),
        )
        for row in existing_rows
    }

    inserted_queries = 0
    inserted_evidence = 0
    for spec in QUERY_SPECS:
        query = spec["query"]
        intent = spec["intent"]
        try:
            results = fetch_europe_pmc(query, args.max_results_per_query)
            append_jsonl(
                paths.research_queries,
                {
                    "recorded_at": utcnow(),
                    "campaign_id": args.campaign_id,
                    "topic_id": args.topic_id,
                    "query": query,
                    "intent": intent,
                    "provider": "europe_pmc",
                    "result_count": len(results),
                },
            )
            inserted_queries += 1
        except Exception as exc:  # pragma: no cover - network failure path
            append_jsonl(
                paths.research_queries,
                {
                    "recorded_at": utcnow(),
                    "campaign_id": args.campaign_id,
                    "topic_id": args.topic_id,
                    "query": query,
                    "intent": intent,
                    "provider": "europe_pmc",
                    "result_count": 0,
                    "error": str(exc),
                },
            )
            inserted_queries += 1
            continue

        for item in results:
            title = str(item.get("title") or "").strip()
            source_url = build_source_url(item)
            if not title or not source_url:
                continue
            key = (title, source_url)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            append_jsonl(
                paths.research_evidence,
                {
                    "recorded_at": utcnow(),
                    "campaign_id": args.campaign_id,
                    "topic_id": args.topic_id,
                    "search_query": query,
                    "search_intent": intent,
                    "source_type": "paper",
                    "source_title": title,
                    "source_url": source_url,
                    "why_it_matters": build_why_it_matters(query, title),
                },
            )
            inserted_evidence += 1

    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "topic_id": args.topic_id,
                "query_rows_added": inserted_queries,
                "evidence_rows_added": inserted_evidence,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
