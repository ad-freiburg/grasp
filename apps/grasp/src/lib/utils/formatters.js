export function prettyJson(data) {
  try {
    const json = JSON.stringify(data, null, 2);
    return typeof json === 'string' ? json : '';
  } catch (error) {
    console.warn('Failed to stringify JSON', error);
    return '';
  }
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

// markdown only nests a bullet list under an enumerated item if the bullets
// are indented by at least the width of the item marker (3 for "1. ", 4 for
// "10. "), but plain text function results indent them by 2; also, a list
// not starting at 1 needs a blank line before it to not continue a paragraph
export function nestBulletsUnderEnumeration(text) {
  const lines = [];
  let width = null;
  for (const line of text.split('\n')) {
    const enumerated = line.match(/^\d+\. /);
    const bullet = line.match(/^ {1,3}(- .*)$/);
    if (enumerated) {
      if (width === null && lines.length > 0 && lines[lines.length - 1].trim()) {
        lines.push('');
      }
      width = enumerated[0].length;
      lines.push(line);
    } else if (bullet && width !== null) {
      lines.push(' '.repeat(width) + bullet[1]);
    } else {
      width = null;
      lines.push(line);
    }
  }
  return lines.join('\n');
}
