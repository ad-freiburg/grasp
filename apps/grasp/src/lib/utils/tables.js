// display of a table value or witness term: { text, href, tag }
export function displayValue(value) {
  if (!value) return { text: '' };
  if (value.type === 'uri') return { text: iriText(value), href: value.value };
  if (value.type === 'bnode') return { text: value.label ?? `_:${value.value}` };
  let text = String(value.value ?? '');
  // dates are often midnight datetimes, show them as dates
  const date = text.match(/^(-?\d{4,}-\d{2}-\d{2})T00:00:00(?:\.0+)?Z$/);
  if (date && value.datatype?.endsWith('#dateTime')) text = date[1];
  if (text.length > 200) text = `${text.slice(0, 200)}…`;
  return { text, tag: value.lang ?? null };
}

function iriText(value) {
  // label and short identifier of table values; table values of older shares
  // and witness terms have the full text as label
  if (!value.short_identifier) return value.label ?? value.value;
  return value.label ? `${value.label} (${value.short_identifier})` : value.short_identifier;
}
