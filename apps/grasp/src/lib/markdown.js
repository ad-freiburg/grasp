import { marked } from 'marked';
import DOMPurify from 'dompurify';
import hljs from 'highlight.js';

marked.setOptions({
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true
});

// raw html in model outputs or notes (e.g. a placeholder like <Title>) is
// shown as text; the browser would otherwise parse it as a tag
const escapeHtml = (text) =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

marked.use({
  renderer: {
    html({ text }) {
      return escapeHtml(text);
    }
  }
});

export function renderMarkdown(content) {
  const dirty = marked.parse(content ?? '');
  return DOMPurify.sanitize(dirty);
}
