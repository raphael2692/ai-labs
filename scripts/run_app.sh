#!/usr/bin/env bash
# Runs one of the apps/ Streamlit apps from the repo root.
# Usage: ./scripts/run_app.sh meeting_minutes
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

app_dir="$1"
if [ ! -f "apps/$app_dir/app.py" ]; then
    echo "No app.py found at apps/$app_dir/app.py"
    echo "Available apps:"
    ls apps
    exit 1
fi

package_name=$(grep -m1 '^name = ' "apps/$app_dir/pyproject.toml" | sed -E 's/name = "(.*)"/\1/')
uv run --package "$package_name" streamlit run "apps/$app_dir/app.py"
