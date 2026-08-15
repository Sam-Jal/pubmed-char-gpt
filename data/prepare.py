"""Build a plain-text corpus from PubMed abstracts through NCBI E-utilities."""

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERY = "medical informatics[MeSH Major Topic] OR oncology[MeSH Major Topic]"
MAX_PUBMED_IDS = 10_000


def _request_params():
    params = {"tool": "pubmed-char-gpt"}
    if email := os.getenv("NCBI_EMAIL"):
        params["email"] = email
    if api_key := os.getenv("NCBI_API_KEY"):
        params["api_key"] = api_key
    return params


def _extract_abstracts(xml_content):
    root = ET.fromstring(xml_content)
    abstracts = []
    for article in root.findall(".//PubmedArticle"):
        sections = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            if not text:
                continue
            if label := node.attrib.get("Label"):
                text = f"{label}: {text}"
            sections.append(text)

        abstract = " ".join(sections)
        if len(abstract) > 100:
            abstracts.append(abstract)
    return abstracts


def fetch_pubmed_abstracts(n_abstracts=9_999, output_path="data/corpus.txt"):
    """Fetch up to 10,000 PubMed records and save one abstract per block."""
    import requests

    if not 1 <= n_abstracts <= MAX_PUBMED_IDS:
        raise ValueError(
            "PubMed ESearch exposes at most the first 10,000 matching IDs. "
            "Choose n_abstracts between 1 and 10,000."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    common_params = _request_params()

    with requests.Session() as session:
        print(f"Fetching up to {n_abstracts} PubMed IDs...")
        response = session.get(
            BASE_URL + "esearch.fcgi",
            params={
                **common_params,
                "db": "pubmed",
                "term": QUERY,
                "retmax": n_abstracts,
                "retmode": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        pmids = response.json()["esearchresult"]["idlist"]
        print(f"Retrieved {len(pmids)} PubMed IDs")

        abstracts = []
        batch_size = 200
        delay = 0.11 if "api_key" in common_params else 0.34
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            print(f"Fetching records {start + 1}-{start + len(batch)}...")
            response = session.get(
                BASE_URL + "efetch.fcgi",
                params={
                    **common_params,
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "abstract",
                    "retmode": "xml",
                },
                timeout=60,
            )
            response.raise_for_status()
            abstracts.extend(_extract_abstracts(response.content))
            time.sleep(delay)

    corpus = "\n\n".join(abstracts)
    output_path.write_text(corpus, encoding="utf-8")
    print(f"Saved {len(abstracts)} abstracts ({len(corpus):,} characters)")
    print(f"Corpus path: {output_path}")


if __name__ == "__main__":
    fetch_pubmed_abstracts()
