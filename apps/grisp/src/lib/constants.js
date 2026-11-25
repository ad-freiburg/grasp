/* global __API_BASE__ __COPYRIGHT__ __MATERIAL_URL__ */

/**
 * Copyright text, set at build time via the COPYRIGHT env var.
 * Defaults to 'University of Freiburg'.
 */
export const COPYRIGHT = __COPYRIGHT__;

/**
 * URL for the "Material" footer link, set at build time via the
 * MATERIAL_URL env var. Defaults to the GRISP page on ad-publications.
 */
export const MATERIAL_URL = __MATERIAL_URL__;

// chair website, linked in the copyright line of the footer
export const CHAIR_URL = 'https://ad.cs.uni-freiburg.de';

// "GitHub" footer link, the GRISP code in the GRASP repository
export const REPO_URL = 'https://github.com/ad-freiburg/grasp/tree/main/src/grasp/baselines/grisp';

/**
 * API base URL, set at build time via the API_BASE env var.
 *
 *   API_BASE=/api                          (default – same origin, reverse proxy)
 *   API_BASE=http://localhost:6790         (direct, dev)
 *   API_BASE=https://example.com/my/api    (custom prefix)
 *
 * Relative paths (including the default /api) are stripped of leading slashes
 * so that the browser resolves them relative to the current page URL.
 * This lets a single build work at any mount point (e.g. "/" and "/grisp/").
 */
const RAW = __API_BASE__.replace(/\/+$/, '');
const isAbsoluteUrl = /^https?:\/\//.test(RAW);
const API_BASE = isAbsoluteUrl ? RAW : RAW.replace(/^\/+/, '');

export const getApiBase = () => API_BASE;

export const wsEndpoint = () => {
  if (isAbsoluteUrl) {
    return API_BASE.replace(/^http/, 'ws') + '/live';
  }
  // Resolve the relative API path against the current page URL
  // to build an absolute WebSocket URL.
  const resolved = new URL(API_BASE, window.location.href);
  const wsProtocol = resolved.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${wsProtocol}//${resolved.host}${resolved.pathname}/live`;
};

export const configEndpoint = () => `${API_BASE}/config`;
export const kgEndpoint = () => `${API_BASE}/knowledge_graphs`;
