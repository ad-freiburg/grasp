// DOM ranges for character spans of the text content of an element, e.g.
// spans the server computed for a SPARQL query shown in the element
export function spanRanges(element, spans) {
  if (!element) return [];
  const nodes = [];
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  let node;
  let offset = 0;
  while ((node = walker.nextNode())) {
    nodes.push([node, offset]);
    offset += node.data.length;
  }
  if (nodes.length === 0) return [];
  const at = (position) => {
    const found = nodes.findLast(([, start]) => start <= position) ?? nodes[0];
    return [found[0], Math.min(position - found[1], found[0].data.length)];
  };
  return spans.map(([start, end]) => {
    const range = document.createRange();
    range.setStart(...at(start));
    range.setEnd(...at(end));
    return range;
  });
}

// highlight character spans of the SPARQL query in a container, e.g. the
// positions of a query item while it is hovered; no spans clear it
export function highlightQuerySpans(container, spans) {
  if (typeof CSS === 'undefined' || !CSS.highlights) return;
  CSS.highlights.delete('query-item');
  const code = container?.querySelector('.sparql-block__code code');
  const ranges = spans?.length ? spanRanges(code, spans) : [];
  if (ranges.length > 0) CSS.highlights.set('query-item', new Highlight(...ranges));
}
