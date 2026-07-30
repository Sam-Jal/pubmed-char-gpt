import os
import requests
import xml.etree.ElementTree as ET

def fetch_pubmed_abstracts(n_abstracts=20000, output_path="data/corpus.txt"):
    """
    Fetches Pubmed abstractions via NCBI E-utilites API ( no account needed).
    two steps:
        esearch ---> get a list of PMIDs matching a query
        efetch -----> retrieve the actual abstractions for thos PMIDs
    """

    Base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    # Broad query - covers most of biomediacl literature
    Query = "medical informatics[MeSH Major Topic] OR oncology[MeSH Major Topic]"
    Batch = 200 #efecth linit per request

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    print(f"Fetching {n_abstracts} PMIDs...") #?
    r = requests.get(Base + "esearch.fcgi" , params={
        "db" : "pubmed",
        "term" : Query,
        "retmax" : n_abstracts, 
        "retmode" : "json",
    })

    r.raise_for_status()
    pmids = r.json()["esearchresult"]["idlist"]
    print(f"Got {len(pmids)} PMIDs")


    all_text = []
    for i in range(0, len(pmids), Batch):
        batch = pmids[i : i + Batch]
        print(f"Fetching abstracts {i}-{i+Batch}...")
        r = requests.get(Base + "efetch.fcgi", params={
            "db" : "pubmed",
            "id" : ",".join(batch),
            "rettype" : "abstract",
            "retmode" : "xml",
        })

        r.raise_for_status()


        # Parse XML - abstracts live at .//AbstractText
        root = ET.fromstring(r.content)
        for node in root.findall(".//AbstractText"):
            text = (node.text or "").strip()
            if len(text) > 100: #??
                all_text.append(text)
        
    corpus = "\n\n".join(all_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nDone. {len(all_text)} abstracts ----> {len(corpus):,} characters")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    fetch_pubmed_abstracts()

