import { useState, useCallback } from 'react';

export interface ZoomState {
  left: string | number | null;
  right: string | number | null;
  refAreaLeft: string | number | null;
  refAreaRight: string | number | null;
  isZoomed: boolean;
}

/**
 * Hook for handling drag-to-zoom functionality in Recharts.
 * Returns state and handlers for zoom selection.
 */
export function useChartZoom() {
  const [zoom, setZoom] = useState<ZoomState>({
    left: null,
    right: null,
    refAreaLeft: null,
    refAreaRight: null,
    isZoomed: false,
  });

  const handleMouseDown = useCallback((e: { activeLabel?: string | number } | null) => {
    if (e?.activeLabel !== undefined) {
      setZoom(z => ({ ...z, refAreaLeft: e.activeLabel! }));
    }
  }, []);

  const handleMouseMove = useCallback((e: { activeLabel?: string | number } | null) => {
    if (zoom.refAreaLeft !== null && e?.activeLabel !== undefined) {
      setZoom(z => ({ ...z, refAreaRight: e.activeLabel! }));
    }
  }, [zoom.refAreaLeft]);

  const handleMouseUp = useCallback(() => {
    if (zoom.refAreaLeft !== null && zoom.refAreaRight !== null) {
      // Sort to ensure left < right
      let left = zoom.refAreaLeft;
      let right = zoom.refAreaRight;
      if (typeof left === 'number' && typeof right === 'number' && left > right) {
        [left, right] = [right, left];
      } else if (typeof left === 'string' && typeof right === 'string' && left > right) {
        [left, right] = [right, left];
      }

      setZoom({
        left,
        right,
        refAreaLeft: null,
        refAreaRight: null,
        isZoomed: true,
      });
    } else {
      setZoom(z => ({ ...z, refAreaLeft: null, refAreaRight: null }));
    }
  }, [zoom.refAreaLeft, zoom.refAreaRight]);

  const resetZoom = useCallback(() => {
    setZoom({
      left: null,
      right: null,
      refAreaLeft: null,
      refAreaRight: null,
      isZoomed: false,
    });
  }, []);

  return { zoom, handleMouseDown, handleMouseMove, handleMouseUp, resetZoom };
}
