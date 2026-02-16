import { useState, useEffect } from 'react';

export function useDimensions(ref, triggers = {}) {
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Update dimensions on window resize
  useEffect(() => {
    const updateDimensions = () => {
      if (ref.current) {
        setDimensions({
          width: ref.current.clientWidth,
          height: ref.current.clientHeight,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [ref]);

  // Recalculate dimensions when external triggers change (e.g. graphData, notes count)
  const { graphData, notesLength } = triggers;
  useEffect(() => {
    if (ref.current && (graphData || notesLength > 0)) {
      const timer = setTimeout(() => {
        if (ref.current) {
          const newWidth = ref.current.clientWidth;
          const newHeight = ref.current.clientHeight;

          setDimensions((prev) => {
            if (prev.width !== newWidth || prev.height !== newHeight) {
              return { width: newWidth, height: newHeight };
            }
            return prev;
          });
        }
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [ref, graphData, notesLength]);

  return dimensions;
}
