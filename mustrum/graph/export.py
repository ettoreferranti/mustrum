"""Graph export (FR-6): ideas as hubs with their sources around them, plus
idea↔idea and contact links, rendered as ONE self-contained HTML file —
Cytoscape.js is embedded, so the file opens offline in any browser."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from mustrum.core.models import EntityKind, MatchStatus
from mustrum.core.ports import StorageRepo


def build_graph_data(repo: StorageRepo, include_contacts: bool = True) -> dict[str, Any]:
    """Cytoscape-format elements: nodes + edges with type/status metadata."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for idea in repo.list_ideas():
        version = repo.latest_idea_version(idea.id)  # type: ignore[arg-type]
        nodes.append(
            {
                "data": {
                    "id": f"idea-{idea.id}",
                    "label": idea.title,
                    "type": "idea",
                    "detail": version.text if version else "",
                    "tags": sorted(repo.tags_for(EntityKind.IDEA, idea.id)),  # type: ignore[arg-type]
                }
            }
        )

    for source in repo.list_sources():
        summary = repo.get_summary(source.id)  # type: ignore[arg-type]
        bib = repo.get_bib_entry(source.id)  # type: ignore[arg-type]
        nodes.append(
            {
                "data": {
                    "id": f"source-{source.id}",
                    "label": source.title,
                    "type": "source",
                    "kind": source.kind.value,
                    "year": source.year,
                    "authors": list(source.authors),
                    "citation_key": bib.citation_key if bib else None,
                    "detail": summary.text if summary else "",
                    "tags": sorted(repo.tags_for(EntityKind.SOURCE, source.id)),  # type: ignore[arg-type]
                }
            }
        )

    for match in repo.list_matches():
        if match.status == MatchStatus.REJECTED:
            continue
        edges.append(
            {
                "data": {
                    "id": f"match-{match.id}",
                    "source": f"idea-{match.idea_id}",
                    "target": f"source-{match.source_id}",
                    "type": "match",
                    "status": match.status.value,
                    "score": round(match.score, 3),
                }
            }
        )

    for i, link in enumerate(repo.list_idea_links()):
        edges.append(
            {
                "data": {
                    "id": f"idealink-{i}",
                    "source": f"idea-{link.from_idea_id}",
                    "target": f"idea-{link.to_idea_id}",
                    "type": "idea-link",
                    "relation": link.relation.value,
                }
            }
        )

    if include_contacts:
        linked_contact_ids = set()
        for i, clink in enumerate(repo.list_contact_links()):
            target = f"idea-{clink.idea_id}" if clink.idea_id else f"source-{clink.source_id}"
            edges.append(
                {
                    "data": {
                        "id": f"contactlink-{i}",
                        "source": f"contact-{clink.contact_id}",
                        "target": target,
                        "type": "contact-link",
                        "why": clink.why,
                    }
                }
            )
            linked_contact_ids.add(clink.contact_id)
        for contact in repo.list_contacts():
            if contact.id in linked_contact_ids:
                nodes.append(
                    {
                        "data": {
                            "id": f"contact-{contact.id}",
                            "label": contact.name,
                            "type": "contact",
                            "kind": contact.kind.value,
                            "detail": contact.affiliation,
                        }
                    }
                )

    return {"nodes": nodes, "edges": edges}


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mustrum — knowledge graph</title>
<style>
  html, body {{ margin: 0; height: 100%; font-family: -apple-system, sans-serif; }}
  #cy {{ position: absolute; inset: 0; }}
  #panel {{ position: absolute; top: 12px; right: 12px; width: 300px; max-height: 60%;
           overflow: auto; background: #ffffffee; border: 1px solid #ccc; border-radius: 8px;
           padding: 12px; font-size: 13px; display: none; z-index: 10; }}
  #legend {{ position: absolute; bottom: 12px; left: 12px; background: #ffffffee;
            border: 1px solid #ccc; border-radius: 8px; padding: 8px 12px; font-size: 12px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 4px; }}
</style>
<script>{cytoscape_js}</script>
</head>
<body>
<div id="cy"></div>
<div id="panel"></div>
<div id="legend">
  <span class="dot" style="background:#f2a33c"></span>idea&nbsp;&nbsp;
  <span class="dot" style="background:#4a90d9"></span>source&nbsp;&nbsp;
  <span class="dot" style="background:#7bb661"></span>contact&nbsp;&nbsp;
  solid = confirmed, dashed = suggested
</div>
<script>
const elements = {elements_json};
const cy = cytoscape({{
  container: document.getElementById("cy"),
  elements: elements,
  style: [
    {{ selector: 'node[type="idea"]',
      style: {{ 'background-color': '#f2a33c', 'label': 'data(label)', 'font-size': 12,
               'text-wrap': 'wrap', 'text-max-width': 140, 'width': 46, 'height': 46 }} }},
    {{ selector: 'node[type="source"]',
      style: {{ 'background-color': '#4a90d9', 'label': 'data(label)', 'font-size': 9,
               'text-wrap': 'wrap', 'text-max-width': 120, 'width': 26, 'height': 26 }} }},
    {{ selector: 'node[type="contact"]',
      style: {{ 'background-color': '#7bb661', 'shape': 'round-rectangle',
               'label': 'data(label)', 'font-size': 10, 'width': 30, 'height': 22 }} }},
    {{ selector: 'edge', style: {{ 'curve-style': 'bezier', 'width': 1.5,
               'line-color': '#bbb' }} }},
    {{ selector: 'edge[status="confirmed"]', style: {{ 'line-color': '#555', 'width': 2.5 }} }},
    {{ selector: 'edge[status="suggested"]', style: {{ 'line-style': 'dashed' }} }},
    {{ selector: 'edge[type="idea-link"]',
      style: {{ 'line-color': '#f2a33c', 'label': 'data(relation)', 'font-size': 8 }} }},
    {{ selector: 'edge[type="contact-link"]',
      style: {{ 'line-color': '#7bb661', 'line-style': 'dotted' }} }}
  ],
  layout: {{ name: 'cose', animate: false, nodeRepulsion: 40000, idealEdgeLength: 90 }}
}});
const panel = document.getElementById("panel");
cy.on('tap', 'node', evt => {{
  const d = evt.target.data();
  let extra = "";
  if (d.type === "source") {{
    extra = (d.authors && d.authors.length ? d.authors.join(", ") : "") +
            (d.year ? " (" + d.year + ")" : "") +
            (d.citation_key ? "<br><code>[@" + d.citation_key + "]</code>" : "");
  }}
  panel.innerHTML = "<b>" + d.label + "</b><br><i>" + d.type +
    (d.kind ? " — " + d.kind : "") + "</i><br>" + extra +
    (d.tags && d.tags.length ? "<br>tags: " + d.tags.join(", ") : "") +
    (d.detail ? "<hr>" + d.detail : "");
  panel.style.display = "block";
}});
cy.on('tap', evt => {{ if (evt.target === cy) panel.style.display = "none"; }});
</script>
</body>
</html>
"""


def render_html(data: dict[str, Any]) -> str:
    cytoscape_js = resources.files("mustrum.graph").joinpath("vendor/cytoscape.min.js").read_text()
    elements = data["nodes"] + data["edges"]
    # </script> inside JSON strings would break out of the script tag
    elements_json = json.dumps(elements).replace("</", "<\\/")
    return _PAGE.format(
        cytoscape_js=cytoscape_js,
        elements_json=elements_json,
    )


def export_graph(repo: StorageRepo, include_contacts: bool = True) -> str:
    return render_html(build_graph_data(repo, include_contacts=include_contacts))
