export function prettyJson(data) {
  try {
    const json = JSON.stringify(data, null, 2);
    return typeof json === 'string' ? json : '';
  } catch (error) {
    console.warn('Failed to stringify JSON', error);
    return '';
  }
}

export function normalizeWhitespace(text) {
  return String(text).replace(/\s+/g, ' ').trim();
}

export function toPreviewText(value) {
  if (value === null || value === undefined) {
    return '';
  }

  if (typeof value === 'string') {
    return normalizeWhitespace(value);
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  try {
    return normalizeWhitespace(JSON.stringify(value));
  } catch (error) {
    console.warn('Failed to stringify function argument', error);
    return normalizeWhitespace(String(value));
  }
}

export function truncatePreview(text, maxLength = 40) {
  const normalized = normalizeWhitespace(text);
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength)}...`;
}

export function flattenFunctionArgs(args, prefix = '') {
  const entries = [];
  if (!args || typeof args !== 'object') {
    return entries;
  }
  for (const [key, value] of Object.entries(args)) {
    if (value === null || value === undefined) continue;
    const nextKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'object' && !Array.isArray(value)) {
      entries.push(...flattenFunctionArgs(value, nextKey));
    } else {
      entries.push({ key: nextKey, value });
    }
  }
  return entries;
}
