import React from 'react';

function Row({ label, example }) {
  return (
    <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem' }}>
      <div style={{ minWidth: 140, color: 'var(--text-muted)', fontWeight: 500 }}>{label}</div>
      <pre style={{
        margin: 0,
        padding: '0.5rem 0.75rem',
        background: 'var(--bg-primary)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        overflowX: 'auto',
        fontSize: '0.8125rem',
        lineHeight: 1.5
      }}>
        <code>{example}</code>
      </pre>
    </div>
  );
}

export default function MarkdownCheatsheet({ isOpen, onClose }) {
  if (!isOpen) return null;
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="md-cheatsheet-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="md-cheatsheet-title">Markdown Cheatsheet</h3>
        </div>

        <div className="modal-body">
          <p style={{ marginBottom: '0.75rem' }}>
            Quick reference for common Markdown syntax supported in the editor and preview.
          </p>

          <Row label="Headers" example={
`# H1
## H2
### H3
#### H4
##### H5
###### H6`
          } />

          <Row label="Bold / Italic" example={
`**bold** or __bold__
*italic* or _italic_
~~strikethrough~~`
          } />

          <Row label="Lists" example={
`- item A
- item B
  - nested
1. First
2. Second`
          } />

          <Row label="Links / Images" example={
`[OpenAI](https://openai.com)
![Alt text](https://placehold.co/200x100)`
          } />

          <Row label="Inline code" example={
"`code` with backticks"
          } />

          <Row label="Code block" example={
"```js\nfunction hello(name) {\n  console.log('Hello ' + name);\n}\n```"
          } />

          <Row label="Blockquote" example={
`> A wise quote
> - Author`
          } />

          <Row label="Tables (GFM)" example={
`| Name | Role  |
|-----:|:-----:|
| Alice| Admin |
| Bob  | User  |`
          } />
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}