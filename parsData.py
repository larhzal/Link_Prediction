import json
import re

author_file = 'aminer_files\AMiner-Author.txt'
coauthor_file = 'aminer_files\AMiner-Coauthor.txt'

def clean_numeric(value, value_type=float, default=0):
    try:
        cleaned = re.sub(r'[^\d.\-]', '', value.strip())
        return value_type(cleaned)
    except Exception as e:
        return default

def parse_authors(path):
    authors = []
    current = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#index"):
                if current:
                    authors.append(current)
                current = {
                    "id": int(line[6:].strip()),
                    "name": "",
                    "affiliations": [],
                    "paper_count": 0,
                    "citation_count": 0,
                    "h_index": 0,
                    "p_index_eq": 0.0,
                    "p_index_uneq": 0.0,
                    "research_interests": []
                }
            elif line.startswith("#n"):
                current["name"] = line[2:].strip()
            elif line.startswith("#a"):
                current["affiliations"] = [a.strip() for a in line[2:].split(';') if a.strip()]
            elif line.startswith("#pc"):
                current["paper_count"] = clean_numeric(line[3:], int)
            elif line.startswith("#cn"):
                current["citation_count"] = clean_numeric(line[3:], int)
            elif line.startswith("#hi"):
                current["h_index"] = clean_numeric(line[3:], int)
            elif line.startswith("#pi"):
                current["p_index_eq"] = clean_numeric(line[3:], float)
            elif line.startswith("#upi"):
                current["p_index_uneq"] = clean_numeric(line[4:], float)
            elif line.startswith("#t"):
                current["research_interests"] = [t.strip() for t in line[2:].split(';') if t.strip()]
        if current:
            authors.append(current)
    return authors

def parse_edges(path):
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                parts = line[1:].strip().split()
                if len(parts) == 3:
                    try:
                        source = int(parts[0])
                        target = int(parts[1])
                        weight = int(parts[2])
                        edges.append({
                            "source": source,
                            "target": target,
                            "weight": weight
                        })
                    except:
                        continue
    return edges

# Traitement
print("Parsing des fichiers...")
authors = parse_authors(author_file)
edges = parse_edges(coauthor_file)

# Sauvegarde JSON
with open("nodes.json", "w", encoding='utf-8') as f:
    json.dump(authors, f, indent=2, ensure_ascii=False)

with open("edges.json", "w", encoding='utf-8') as f:
    json.dump(edges, f, indent=2, ensure_ascii=False)

print("Fichiers 'nodes.json' et 'edges.json' générés avec succès.")
