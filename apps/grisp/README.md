# GRISP Website (SvelteKit)

SvelteKit web interface for the GRISP (Grammar-Regulated Interactive SPARQL generation with Phrase alternatives) baseline. Provides a simple input field, progress indicator, output card, and expandable intermediate steps (skeletons, selection events).

## Requirements

- Node.js 20.19+ (or 22.12+/24+) — the tooling enforces this.
  We recommend using `nvm` to manage versions.
- npm (ships with Node).

Install project dependencies:

```bash
cd apps/grisp
npm install
```

This installs all required dependencies including markdown rendering libraries (marked, dompurify, highlight.js).

## Configuring Examples

Example questions are defined in `src/lib/examples.json`. You can customize these by editing the JSON file:

```json
[
  {
    "label": "Short descriptive label",
    "question": "The full question text that will be inserted"
  }
]
```

The examples dropdown will automatically populate with these entries.

## Development

```bash
API_BASE=http://localhost:6790 npm run dev
```

The dev server runs at http://localhost:5173 with hot module replacement. `API_BASE` points the website at a running GRISP server.

## Building for production

```bash
npm run build
```

With the static adapter, the production build is emitted to `build/`. You can preview the output locally with:

```bash
npm run preview
```

### Build-time environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE` | `/api` | API base URL. Relative paths are resolved against the current page URL at runtime. Set to an absolute URL (e.g. `http://localhost:6790`) to talk directly to a GRISP server. |
| `COPYRIGHT` | `University of Freiburg` | Copyright holder text displayed in the footer. |
| `MATERIAL_URL` | `grisp` | URL of the "Material" footer link. Relative paths are resolved against the current page URL. |

## Docker

```bash
docker build -t grisp-website .
docker run -p 8080:80 grisp-website
```

Override build args as needed:

```bash
docker build -t grisp-website \
  --build-arg API_BASE=/api \
  --build-arg COPYRIGHT="Your Organization" \
  .
```

The multi-stage Dockerfile compiles the static build using Node 22 Alpine, then serves the exported site via `nginx:alpine-slim`.
