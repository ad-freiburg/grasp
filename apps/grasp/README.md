# GRASP Website (SvelteKit)

SvelteKit port of the GRASP conversational interface. The project mirrors the existing Flutter-based web app while relying on semantic HTML, accessible styling, and a lightweight Svelte component layer.

## Requirements

- Node.js 20.19+ (or 22.12+/24+) — the tooling enforces this.
  We recommend using `nvm` to manage versions.
- npm (ships with Node).

Install project dependencies once:

```bash
cd apps/grasp
npm install
```

## Development

```bash
API_BASE=http://localhost:6789 npm run dev
```

The dev server runs at http://localhost:5173 with hot module replacement. `API_BASE` points the website at a running GRASP server. Source code lives under `src/` with shared components in `src/lib/`. Static assets (favicons, robots.txt, etc.) reside in `static/`.

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
| `BASE_PATH` | `""` | SvelteKit path prefix (e.g. `/grasp`) for when the site is hosted under a subpath. |
| `API_BASE` | `/api` | API base URL. Relative paths are prefixed with `BASE_PATH` (e.g. `BASE_PATH=/v1` + `API_BASE=/api` → `/v1/api`). Set to an absolute URL (e.g. `http://localhost:6789`) to talk directly to a GRASP server. |
| `QLEVER_HOSTS` | `qlever.cs.uni-freiburg.de,qlever.informatik.uni-freiburg.de,qlever.dev,*.qlever.dev` | Comma-separated hosts whose SPARQL endpoint gets an "Execute on QLever" button below a generated query. An entry is either an exact host (`qlever.dev`) or a wildcard suffix (`*.qlever.dev`, matching any subdomain but not the bare domain). Set to an empty string to hide the button everywhere. |

## Docker

```bash
docker build -t grasp-website .
docker run -p 8080:80 grasp-website
```

Override build args as needed:

```bash
docker build -t grasp-website \
  --build-arg BASE_PATH=/grasp \
  --build-arg API_BASE=/api \
  --build-arg QLEVER_HOSTS=qlever.cs.uni-freiburg.de,*.qlever.dev \
  .
```

The multi-stage Dockerfile compiles the static build using Node 22 Alpine, then serves the exported site via `nginx:alpine-slim`.
