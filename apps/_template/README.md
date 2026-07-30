# App Template

Copy this whole folder to `apps/<your_app_name>/` to start a new app:

```
cp -r apps/_template apps/my_new_app
```

Then:

1. Rename `name` in `pyproject.toml` to something unique (e.g. `my-new-app`).
2. Edit `app.py`.
3. Run it: `uv run --package my-new-app streamlit run apps/my_new_app/app.py`.

It already depends on `ai-lab-common` for the shared `Settings`, sidebar, Whisper client, and LLM client —
see `apps/common/README.md`.
