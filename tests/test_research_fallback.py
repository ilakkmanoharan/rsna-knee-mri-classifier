"""Research stage should still retrieve papers when arXiv 406s."""

from __future__ import annotations

import json

from agent.stages import research as research_mod


def test_europepmc_search_parses_lite_results(monkeypatch):
    payload = {
        "resultList": {
            "result": [
                {
                    "title": "MRNet plane-wise knee MRI",
                    "authorString": "Bien et al.",
                    "pubYear": "2018",
                    "doi": "10.1371/journal.pmed.1002699",
                    "source": "MED",
                    "pmid": "123",
                }
            ]
        }
    }
    monkeypatch.setattr(research_mod, "_http_get", lambda url, timeout=45: json.dumps(payload).encode())
    hits = research_mod._europepmc_search("knee MRI", max_results=1)
    assert hits[0]["title"].startswith("MRNet")
    assert hits[0]["url"].startswith("https://doi.org/")
    assert hits[0]["source"] == "europepmc"


def test_run_research_falls_back_to_europepmc(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("HTTP Error 406: Not Acceptable")

    def fake_epmc(query, max_results=2):
        return [
            {
                "title": f"Paper for {query}",
                "summary": "authors",
                "url": "https://doi.org/10.0/test",
                "published": "2025",
                "source": "europepmc",
            }
        ]

    monkeypatch.setattr(research_mod, "_arxiv_search", boom)
    monkeypatch.setattr(research_mod, "_europepmc_search", fake_epmc)
    md = research_mod.run_research(
        tmp_path,
        queries=["knee MRI", "RSNA knee", "macro ROC-AUC"],
        max_arxiv=6,
        cycle_id="00_test",
        day_id="2026-09-20",
    )
    text = md.read_text()
    assert "Paper for knee MRI" in text
    meta = json.loads((tmp_path / "2026-09-20_cycle00_test_research.json").read_text())
    assert len(meta["papers"]) >= 3
    assert any("406" in e for e in meta["errors"])


def test_run_research_seeds_curated_dois_when_both_apis_fail(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("HTTP Error 406: Not Acceptable")

    def epmc_down(*_a, **_k):
        raise RuntimeError("HTTP Error 503: Service Unavailable")

    monkeypatch.setattr(research_mod, "_arxiv_search", boom)
    monkeypatch.setattr(research_mod, "_europepmc_search", epmc_down)
    md = research_mod.run_research(
        tmp_path,
        queries=["knee MRI"],
        max_arxiv=2,
        cycle_id="00_test",
        day_id="2026-09-25",
    )
    meta = json.loads((tmp_path / "2026-09-25_cycle00_test_research.json").read_text())
    assert len(meta["papers"]) >= 3
    assert all(p.get("source") == "curated" for p in meta["papers"])
    assert "doi.org" in md.read_text()
