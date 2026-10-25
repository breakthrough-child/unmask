#!/usr/bin/env bash
# Render the Mermaid blocks in docs/architecture.md and docs/agent-workflow.md to SVG + PNG.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/diagrams
for name in architecture agent-workflow; do
  awk '/^```mermaid/{f=1;next} /^```/{if(f){exit}} f' "docs/$name.md" > "docs/diagrams/$name.mmd"
  npx -y @mermaid-js/mermaid-cli@11.4.2 -i "docs/diagrams/$name.mmd" -o "docs/diagrams/$name.svg" -b white
  npx -y @mermaid-js/mermaid-cli@11.4.2 -i "docs/diagrams/$name.mmd" -o "docs/diagrams/$name.png" -b white -s 2
done
echo "wrote docs/diagrams/*.svg|png"
