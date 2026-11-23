# Deploying

The `Dockerfile` builds the web app with OpenFOAM v1912 (the Ubuntu 24.04 package, the version
the CI tests and the validation use), so cases run from the browser without installing
anything. code_saturne is not included; the app then offers OpenFOAM only.

## Run locally

```bash
docker build -t cfd-case-generator .
docker run --rm -p 8501:8501 cfd-case-generator              # → http://localhost:8501
```

Pass API keys with `-e`, e.g. `-e GEMINI_API_KEY=…` or `-e ANTHROPIC_API_KEY=…`; the sidebar
preselects the provider whose key it finds. Without a key, visitors paste their own.

The same image runs the command-line tool in the OpenFOAM environment:

```bash
docker run --rm -v "$PWD/out:/out" cfd-case-generator \
    python -m app.cli generate examples/dam_break.json -o /out/dambreak --run
```

(The container runs as uid 1000; on Linux, `out/` must be writable by that user.)

## Settings

| Variable | Default | Effect |
|---|---|---|
| `CFD_RUN_TIMEOUT` | `1800` | Wall-clock limit of one solver run, in seconds. A run that reaches it is stopped and reported; the repair agent does not retry it. |
| `PORT` | `8501` | Port the web app listens on. |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | | Provider keys used when the visitor leaves the key field empty. |
| `GEMINI_MODEL`, `GROQ_MODEL`, `CFD_CLAUDE_MODEL` | | Default model of each provider. |

## Online demo on Hugging Face Spaces

A free CPU Space (2 vCPU, 16 GB) runs the quick checks in seconds and the 2-D cases in minutes.

1. On huggingface.co: **New Space**, SDK **Docker** (blank template), hardware **CPU basic**.
2. Push this repository to the Space, with a Space README in place of the GitHub one (a Space
   reads its settings from the README's front matter):

   ```bash
   git clone https://huggingface.co/spaces/<you>/<space> ../space
   git archive HEAD | tar -x -C ../space
   cd ../space
   cat > README.md <<'MD'
   ---
   title: CFD Case Generator
   emoji: 🌊
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 8501
   pinned: false
   short_description: Plain-English flow problems to OpenFOAM cases, run live
   ---
   Describe a flow in plain English; get an OpenFOAM case, run it and check the result.
   Source code: https://github.com/<you>/cfd-case-generator
   MD
   git add -A && git commit -m "Deploy" && git push
   ```

   (A Space rejects files over 10 MB without Git LFS; the GIFs in `docs/` are smaller, but you
   can also delete `docs/` before committing, since the image does not use it.)
3. In the Space **Settings → Variables and secrets**:
   - `CFD_RUN_TIMEOUT` = `600` (variable) keeps one visitor from holding the CPU for long.
   - An API key as a **secret** only if you accept that every visitor uses it (and its
     quota). Without one, visitors paste their own key, which stays in their session.

The Space builds the image (about 5 minutes), then sleeps after 48 hours without visitors
and wakes on the next visit. Each visitor's run lives in their browser session: a page reload
during a run loses the live monitor (the run itself is stopped at `CFD_RUN_TIMEOUT`).
