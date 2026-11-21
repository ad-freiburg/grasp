// statuses of items, for the query of the output and the arguments of tool
// calls
export const STATUS = {
  retrieved: { text: 'surfaced before use', title: 'A tool result showed this item before it was used' },
  verified: { text: 'surfaced after use', title: 'This item was used before a tool result showed it' },
  noted: { text: 'from notes', title: 'No tool result showed this item, but the notes contain it' },
  unverified: { text: 'never surfaced', title: 'No tool result showed this item, it may be made up' }
};

// the arguments of a call shortened for one row, without the knowledge graph
export function callArgs(args) {
  return Object.entries(args ?? {})
    .filter(([key]) => key !== 'kg')
    .map(([key, value]) => {
      let text = typeof value === 'string' ? value : JSON.stringify(value);
      text = String(text).replace(/\s+/g, ' ').trim();
      return [key, text.length > 60 ? `${text.slice(0, 60)}…` : text];
    });
}

// the line of a query containing a position, e.g. of an item in it
export function queryLine(sparql, position) {
  const start = sparql.lastIndexOf('\n', position - 1) + 1;
  const end = sparql.indexOf('\n', position);
  return sparql.slice(start, end < 0 ? undefined : end).trim();
}
