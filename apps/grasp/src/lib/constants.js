/* global __API_BASE__ */

export const APP_COLORS = Object.freeze({
  uniBlue: '#344A9A',
  uniDarkBlue: '#000149',
  uniRed: '#C1002A',
  uniGray: '#B4B4B4',
  uniGreen: '#00A082',
  uniYellow: '#BEAA3C',
  uniPink: '#A35394',
  surface: '#FFFFFF'
});

export const BRAND_LINKS = Object.freeze({
  chair: 'https://ad.cs.uni-freiburg.de',
  repo: 'https://github.com/ad-freiburg/grasp',
  methodPaper: 'https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_WB_2025.pdf',
  systemPaper: 'https://ad-publications.cs.uni-freiburg.de/ISWC_grasp_demo_WB_2025.pdf',
  entityLinkingPaper:
    'https://ad-publications.cs.uni-freiburg.de/SEMTAB_entity_linking_grasp_WB_2025.pdf',
  evaluation: 'evaluate',
  data: 'https://ad-publications.cs.uni-freiburg.de/grasp/'
});

/**
 * API base URL, set at build time via the API_BASE env var.
 *
 *   API_BASE=/api                          (default – same origin, reverse proxy)
 *   API_BASE=http://localhost:6789         (direct, dev)
 *   API_BASE=https://example.com/my/api    (custom prefix)
 *
 * Relative paths (including the default /api) are stripped of leading slashes
 * so that the browser resolves them relative to the current page URL.
 * This lets a single build work at any mount point (e.g. "/" and "/test/").
 */
const RAW = __API_BASE__.replace(/\/+$/, '');
const isAbsoluteUrl = /^https?:\/\//.test(RAW);
const API_BASE = isAbsoluteUrl ? RAW : RAW.replace(/^\/+/, '');

export const getApiBase = () => API_BASE;

export const TASKS = Object.freeze([
  {
    id: 'sparql-qa',
    name: 'SPARQL QA',
    tooltip:
      'Answer questions by generating a corresponding SPARQL query over one or more knowledge graphs.'
  },
  {
    id: 'general-qa',
    name: 'General QA',
    tooltip:
      'Answer questions by retrieving relevant information from knowledge graphs.'
  },
  {
    id: 'sparql-to-question',
    name: 'SPARQL to Question',
    tooltip: 'Convert a SPARQL query into a natural language question.'
  },
  {
    id: 'entity-linking',
    name: 'Entity Linking',
    tooltip:
      'Annotate entity mentions in a text with corresponding knowledge graph entities.'
  },
  {
    id: 'cea',
    name: 'Cell Entity Annotation',
    tooltip:
      'Upload a CSV table to annotate each cell with corresponding knowledge graph entities.'
  }
]);

// tasks that take natural language input and thus support speech-to-text
export const STT_TASKS = Object.freeze(['sparql-qa', 'general-qa', 'entity-linking']);

export const QLEVER_HOSTS = Object.freeze([
  'qlever.cs.uni-freiburg.de',
  'qlever.informatik.uni-freiburg.de',
  'qlever.dev'
]);

// A relative API base (e.g. "api") is resolved against the current document URL.
// The app is only ever served at the mount root — "/", "/?share=:id", or a
// single-segment KG path like "/wikidata" — so the document's directory is
// always the app root, and relative API paths resolve correctly at any path
// prefix depth with no <base> tag involved. (Pretty /share/:id links 302-redirect
// to /?share=:id before the app boots; serving the app one level deep would
// break Safari, which ignores an injected <base> for dynamically imported
// modules — see nginx.conf.)

export const endpointFor = (path) => {
  if (isAbsoluteUrl || typeof window === 'undefined') {
    return `${API_BASE}${path}`;
  }
  return new URL(`${API_BASE}${path}`, window.location.href).href;
};

export const wsEndpoint = () => {
  if (isAbsoluteUrl) {
    return API_BASE.replace(/^http/, 'ws') + '/live';
  }
  const resolved = new URL(API_BASE, window.location.href);
  const wsProtocol = resolved.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${resolved.host}${resolved.pathname}/live`;
};

export const configEndpoint = () => endpointFor('/config');
export const kgEndpoint = () => endpointFor('/knowledge_graphs');
export const transcribeEndpoint = () => endpointFor('/transcribe');
export const saveSharedStateEndpoint = () => endpointFor('/save');
export const loadSharedStateEndpoint = (id) => endpointFor(`/load/${encodeURIComponent(id)}`);
export const sharePathForId = (id) => {
  const trimmed = typeof id === 'string' ? id.trim() : '';
  if (!trimmed) return '';
  if (typeof window === 'undefined') return '';
  // Generate /share/:id path — in production nginx serves index.html for this
  // path (URL unchanged) and the app reads the id from window.location.pathname.
  const base = window.location.pathname.replace(/\/+$/, '');
  return `${window.location.origin}${base}/share/${trimmed}`;
};
