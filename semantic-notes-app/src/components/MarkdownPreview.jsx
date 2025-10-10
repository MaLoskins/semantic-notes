import React, { forwardRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

const MarkdownPreview = forwardRef(function MarkdownPreview({ content, className = '', style, ...divProps }, ref) {
  return (
    <div ref={ref} className={`markdown-preview ${className}`} style={style} {...divProps}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
          code({ node, inline, className, children, ...props }) {
            const text = String(children).replace(/\n$/, '');
            if (inline) {
              return (
                <code className={`inline-code ${className || ''}`.trim()} {...props}>
                  {text}
                </code>
              );
            }
            return (
              <pre className={`code-block ${className || ''}`.trim()}>
                <code {...props}>{text}</code>
              </pre>
            );
          },
          img: ({ node, ...props }) => <img {...props} loading="lazy" />,
        }}
      >
        {content || ''}
      </ReactMarkdown>
    </div>
  );
});

export default MarkdownPreview;