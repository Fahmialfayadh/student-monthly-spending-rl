const ZONE_COLORS = {
  home: 'var(--graph-home)',
  campus: 'var(--graph-campus)',
  third_place: 'var(--graph-third-place)',
  commercial: 'var(--graph-commercial)',
  custom: 'var(--graph-custom)',
};

export const WORLD_UNITS_PER_KM = 100;

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

export function computeWorldDistanceKm(posA, posB) {
  const dx = Number(posB?.x ?? 0) - Number(posA?.x ?? 0);
  const dy = Number(posB?.y ?? 0) - Number(posA?.y ?? 0);
  const km = Math.hypot(dx, dy) / WORLD_UNITS_PER_KM;
  return Math.max(0.1, Math.round(km * 10) / 10);
}

export function ensureWorldPositions(worldConfig) {
  const nodes = worldConfig?.world_graph?.locations || [];
  if (nodes.length === 0) return;
  const maxAbs = Math.max(
    0,
    ...nodes.map((n) => Math.max(Math.abs(Number(n.ui_position?.x ?? 0)), Math.abs(Number(n.ui_position?.y ?? 0)))),
  );
  if (maxAbs > 0 && maxAbs <= 12) {
    nodes.forEach((n) => {
      n.ui_position = {
        x: Math.round(110 + Number(n.ui_position?.x ?? 0) * 60),
        y: Math.round(80 + Number(n.ui_position?.y ?? 0) * 52),
      };
    });
  } else {
    nodes.forEach((n, i) => {
      if (!n.ui_position || !Number.isFinite(Number(n.ui_position.x)) || !Number.isFinite(Number(n.ui_position.y))) {
        n.ui_position = { x: 160 + i * 160, y: 240 };
      }
    });
  }
}

export function createSpatialGraphViewport(svg, hooks = {}) {
  const view = { x: 0, y: 0, w: 820, h: 540 };
  let lastConfig = null;
  let lastOptions = {};
  let panning = false;
  let panMoved = false;
  let panStartClient = { x: 0, y: 0 };
  let panStartView = { x: 0, y: 0 };

  function getSvgRect() {
    const rect = svg.getBoundingClientRect();
    return {
      left: rect.left || 0,
      top: rect.top || 0,
      width: rect.width > 10 ? rect.width : 820,
      height: rect.height > 10 ? rect.height : 540,
    };
  }

  function worldPerScreenPx() {
    const rect = getSvgRect();
    return view.w / rect.width;
  }

  function screenToWorld(clientX, clientY) {
    const rect = getSvgRect();
    return {
      x: view.x + ((clientX - rect.left) / rect.width) * view.w,
      y: view.y + ((clientY - rect.top) / rect.height) * view.h,
    };
  }

  function getScaleBarInfo() {
    const pxPerWorld = 1 / Math.max(1e-6, worldPerScreenPx());
    const pxPerKm = pxPerWorld * WORLD_UNITS_PER_KM;
    const candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
    let bestKm = 1;
    let bestDiff = Infinity;
    for (const km of candidates) {
      const wPx = km * pxPerKm;
      const diff = Math.abs(wPx - 96);
      if (wPx >= 42 && wPx <= 170 && diff < bestDiff) {
        bestDiff = diff;
        bestKm = km;
      }
    }
    const widthPx = Math.max(32, Math.min(180, Math.round(bestKm * pxPerKm)));
    const label = bestKm < 1 ? `${Math.round(bestKm * 1000)} m` : `${bestKm} km`;
    return { widthPx, label };
  }

  function applyViewBox(rerender = false) {
    svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
    if (hooks.onCameraChange) hooks.onCameraChange(getScaleBarInfo());
    if (rerender && lastConfig) render(lastConfig, lastOptions);
  }

  function syncAspectToRect() {
    const rect = getSvgRect();
    const aspect = rect.height / rect.width;
    const cy = view.y + view.h / 2;
    view.h = view.w * aspect;
    view.y = cy - view.h / 2;
  }

  function fitToNodes(worldConfig) {
    const cfg = worldConfig || lastConfig;
    const nodes = cfg?.world_graph?.locations || [];
    const rect = getSvgRect();
    const aspect = rect.height / rect.width;

    if (nodes.length === 0) {
      view.x = 0;
      view.y = 0;
      view.w = 820;
      view.h = 820 * aspect;
      applyViewBox(true);
      return;
    }

    ensureWorldPositions(cfg);
    const xs = nodes.map((n) => Number(n.ui_position.x));
    const ys = nodes.map((n) => Number(n.ui_position.y));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const spanX = Math.max(260, maxX - minX);
    const spanY = Math.max(200, maxY - minY);
    const padX = Math.max(110, spanX * 0.28);
    const padY = Math.max(100, spanY * 0.3);

    let targetW = spanX + padX * 2;
    let targetH = spanY + padY * 2;
    if (targetH / targetW > aspect) {
      targetW = targetH / aspect;
    } else {
      targetH = targetW * aspect;
    }

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    view.w = Math.max(260, Math.min(20000, targetW));
    view.h = view.w * aspect;
    view.x = cx - view.w / 2;
    view.y = cy - view.h / 2;
    applyViewBox(true);
  }

  function zoomAround(factor, worldAnchorX, worldAnchorY) {
    const rect = getSvgRect();
    const aspect = rect.height / rect.width;
    const nextW = Math.max(160, Math.min(20000, view.w * factor));
    const actualFactor = nextW / view.w;
    view.x = worldAnchorX - (worldAnchorX - view.x) * actualFactor;
    view.y = worldAnchorY - (worldAnchorY - view.y) * actualFactor;
    view.w = nextW;
    view.h = nextW * aspect;
    applyViewBox(true);
  }

  function zoomBy(factor) {
    zoomAround(factor, view.x + view.w / 2, view.y + view.h / 2);
  }

  function getCenterWorldPos() {
    return {
      x: Math.round(view.x + view.w / 2),
      y: Math.round(view.y + view.h / 2),
    };
  }

  // Pan on empty canvas drag
  svg.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0 && ev.button !== 1) return;
    if (ev.target.closest('[data-graph-node]') || ev.target.closest('[data-graph-edge]')) return;
    panning = true;
    panMoved = false;
    panStartClient = { x: ev.clientX, y: ev.clientY };
    panStartView = { x: view.x, y: view.y };
    svg.style.cursor = 'grabbing';
    svg.setPointerCapture(ev.pointerId);
  });

  svg.addEventListener('pointermove', (ev) => {
    if (!panning) return;
    const dxPx = ev.clientX - panStartClient.x;
    const dyPx = ev.clientY - panStartClient.y;
    if (!panMoved && Math.hypot(dxPx, dyPx) < 4) return;
    panMoved = true;
    const rect = getSvgRect();
    view.x = panStartView.x - (dxPx / rect.width) * view.w;
    view.y = panStartView.y - (dyPx / rect.height) * view.h;
    applyViewBox(false);
  });

  const endPan = (ev) => {
    if (!panning) return;
    panning = false;
    svg.style.cursor = 'grab';
    if (svg.hasPointerCapture(ev.pointerId)) svg.releasePointerCapture(ev.pointerId);
    if (!panMoved && lastOptions.onCanvasClick) {
      lastOptions.onCanvasClick();
    }
  };
  svg.addEventListener('pointerup', endPan);
  svg.addEventListener('pointercancel', endPan);

  // Smooth zoom around cursor via wheel
  svg.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const anchor = screenToWorld(ev.clientX, ev.clientY);
    const factor = Math.exp(Math.max(-240, Math.min(240, ev.deltaY)) * 0.0014);
    zoomAround(factor, anchor.x, anchor.y);
  }, { passive: false });

  svg.style.cursor = 'grab';

  function render(worldConfig, options = {}) {
    lastConfig = worldConfig;
    lastOptions = options;
    const {
      interactive = true,
      selectedItem = null,
      connectSourceId = null,
      highlightedEdgeIds = [],
      highlightMode = 'fastest',
      agentNodeId = null,
      onSelect = null,
      onNodeMove = null,
    } = options;

    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!worldConfig?.world_graph) return;

    ensureWorldPositions(worldConfig);
    syncAspectToRect();
    svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);

    // Keep visual sizes crisp across all zoom levels (like Google Maps / Figma)
    const s = Math.max(0.65, Math.min(7.5, worldPerScreenPx()));

    const defs = svgEl('defs');
    const marker = svgEl('marker', {
      id: `oneway-arrow-${svg.id || 'g'}`,
      viewBox: '0 0 10 10',
      refX: '24',
      refY: '5',
      markerWidth: '6',
      markerHeight: '6',
      orient: 'auto-start-reverse',
    });
    marker.appendChild(svgEl('path', { d: 'M 0 1 L 10 5 L 0 9 z', fill: 'var(--ink-secondary)' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    const nodes = worldConfig.world_graph.locations || [];
    const edges = worldConfig.world_graph.edges || [];
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const highlighted = new Set(highlightedEdgeIds);

    const edgeVisuals = [];
    const pairCounts = new Map();

    edges.forEach((edge) => {
      const src = nodeMap.get(edge.source);
      const dst = nodeMap.get(edge.target);
      if (!src || !dst || edge.source === edge.target) return;

      const key = [edge.source, edge.target].sort().join('::');
      const idx = pairCounts.get(key) || 0;
      pairCounts.set(key, idx + 1);
      const offset = idx === 0 ? 0 : (idx % 2 === 1 ? 1 : -1) * Math.ceil(idx / 2) * (24 * s);

      const isSelected = selectedItem?.type === 'edge' && selectedItem.id === edge.id;
      const isHighlighted = highlighted.has(edge.id);

      const stroke = isHighlighted
        ? (highlightMode === 'cheapest' ? 'var(--graph-route-cheap)' : 'var(--graph-route-fast)')
        : (isSelected ? 'var(--ink-primary)' : 'var(--border-strong)');

      const pathAttrs = {
        fill: 'none',
        stroke,
        'stroke-width': (isHighlighted || isSelected ? 2.8 : 1.75) * s,
        'stroke-dasharray': edge.mode === 'walk' ? `${6 * s} ${4 * s}` : 'none',
        'data-graph-edge': edge.id,
        style: interactive ? 'cursor: pointer;' : '',
      };
      if (edge.bidirectional === false) {
        pathAttrs['marker-end'] = `url(#oneway-arrow-${svg.id || 'g'})`;
      }

      const path = svgEl('path', pathAttrs);
      const hitPath = svgEl('path', {
        fill: 'none',
        stroke: 'transparent',
        'stroke-width': 16 * s,
        'data-graph-edge': edge.id,
        style: interactive ? 'cursor: pointer;' : '',
      });

      const labelText = svgEl('text', {
        'text-anchor': 'middle',
        fill: isHighlighted ? stroke : 'var(--ink-secondary)',
        'font-size': 10.5 * s,
        'font-family': 'var(--font-sans)',
        'font-weight': isHighlighted || isSelected ? 700 : 600,
        style: `pointer-events: none; paint-order: stroke; stroke: var(--bg-surface); stroke-width: ${4 * s}px; stroke-linejoin: round;`,
      });

      if (interactive && onSelect) {
        hitPath.addEventListener('click', (ev) => {
          ev.stopPropagation();
          onSelect({ type: 'edge', id: edge.id });
        });
      }

      svg.appendChild(path);
      svg.appendChild(hitPath);
      svg.appendChild(labelText);

      edgeVisuals.push({ edge, offset, path, hitPath, labelText });
    });

    function updateEdgeGeometry() {
      edgeVisuals.forEach(({ edge, offset, path, hitPath, labelText }) => {
        const src = nodeMap.get(edge.source);
        const dst = nodeMap.get(edge.target);
        if (!src || !dst) return;
        const p1 = src.ui_position;
        const p2 = dst.ui_position;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len;
        const ny = dx / len;
        const cx = (p1.x + p2.x) / 2 + nx * offset;
        const cy = (p1.y + p2.y) / 2 + ny * offset;
        const d = `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`;
        path.setAttribute('d', d);
        hitPath.setAttribute('d', d);

        const mx = 0.25 * p1.x + 0.5 * cx + 0.25 * p2.x;
        const my = 0.25 * p1.y + 0.5 * cy + 0.25 * p2.y;
        labelText.setAttribute('x', String(mx));
        labelText.setAttribute('y', String(my - 4 * s));
        const dirMark = edge.bidirectional === false ? ' →' : '';
        labelText.textContent = `${Number(edge.distance_km).toFixed(1)} km · ${edge.mode}${dirMark}`;
      });
    }

    updateEdgeGeometry();

    nodes.forEach((node) => {
      const pos = node.ui_position;
      const isSelected = selectedItem?.type === 'node' && selectedItem.id === node.id;
      const isConnectSrc = connectSourceId === node.id;
      const isAgentHere = agentNodeId === node.id;
      const isHome = worldConfig.world_graph.home_node_id === node.id;
      const zoneColor = ZONE_COLORS[node.zone] || ZONE_COLORS.custom;

      const g = svgEl('g', {
        transform: `translate(${pos.x}, ${pos.y})`,
        'data-graph-node': node.id,
        style: interactive ? 'cursor: grab;' : '',
      });

      if (isAgentHere) {
        g.appendChild(svgEl('circle', {
          r: 27 * s,
          fill: 'none',
          stroke: 'var(--graph-agent)',
          'stroke-width': 2.5 * s,
        }));
      }

      if (isConnectSrc) {
        g.appendChild(svgEl('circle', {
          r: 26 * s,
          fill: 'none',
          stroke: 'var(--ink-primary)',
          'stroke-width': 2 * s,
          'stroke-dasharray': `${4 * s} ${3 * s}`,
        }));
      }

      g.appendChild(svgEl('circle', {
        r: (isSelected ? 21 : 19) * s,
        fill: 'var(--bg-surface)',
        stroke: zoneColor,
        'stroke-width': (isSelected ? 3.5 : 2.25) * s,
      }));

      g.appendChild(svgEl('circle', {
        r: 6.5 * s,
        fill: zoneColor,
      }));

      const title = svgEl('text', {
        y: 33 * s,
        'text-anchor': 'middle',
        fill: 'var(--ink-primary)',
        'font-size': 11.5 * s,
        'font-weight': 700,
        'font-family': 'var(--font-sans)',
        style: `paint-order: stroke; stroke: var(--bg-surface); stroke-width: ${3.5 * s}px;`,
      });
      const displayLabel = node.label || node.id;
      title.textContent = isHome ? `${displayLabel} (Home)` : displayLabel;
      g.appendChild(title);

      const caps = [
        node.can_sleep && !isHome ? 'sleep' : null,
        node.has_kitchen ? 'dapur' : null,
        node.has_grocery ? 'belanja' : null,
        node.is_class_venue ? 'kelas' : null,
        node.can_study ? 'belajar' : null,
        node.zone === 'third_place' ? 'kafe' : null,
      ].filter(Boolean).slice(0, 4).join(' · ');

      if (caps) {
        const sub = svgEl('text', {
          y: 46 * s,
          'text-anchor': 'middle',
          fill: 'var(--ink-muted)',
          'font-size': 9.5 * s,
          'font-weight': 600,
          'font-family': 'var(--font-sans)',
          style: `paint-order: stroke; stroke: var(--bg-surface); stroke-width: ${2.5 * s}px;`,
        });
        sub.textContent = caps;
        g.appendChild(sub);
      }

      if (interactive) {
        let dragging = false;
        let moved = false;
        let startClientX = 0;
        let startClientY = 0;

        g.addEventListener('pointerdown', (ev) => {
          ev.stopPropagation();
          dragging = true;
          moved = false;
          startClientX = ev.clientX;
          startClientY = ev.clientY;
          g.style.cursor = 'grabbing';
          g.setPointerCapture(ev.pointerId);
        });

        g.addEventListener('pointermove', (ev) => {
          if (!dragging) return;
          if (!moved && Math.hypot(ev.clientX - startClientX, ev.clientY - startClientY) < 4) return;
          moved = true;
          const worldPt = screenToWorld(ev.clientX, ev.clientY);
          const wx = Math.round(worldPt.x);
          const wy = Math.round(worldPt.y);
          node.ui_position = { x: wx, y: wy };
          g.setAttribute('transform', `translate(${wx}, ${wy})`);

          edges.forEach((edge) => {
            if (edge.source === node.id || edge.target === node.id) {
              const other = nodeMap.get(edge.source === node.id ? edge.target : edge.source);
              if (other) {
                edge.distance_km = computeWorldDistanceKm(node.ui_position, other.ui_position);
              }
            }
          });
          updateEdgeGeometry();
          if (onNodeMove) onNodeMove(node.id, wx, wy, false);
        });

        g.addEventListener('pointerup', (ev) => {
          if (!dragging) return;
          dragging = false;
          g.style.cursor = 'grab';
          if (g.hasPointerCapture(ev.pointerId)) g.releasePointerCapture(ev.pointerId);
          if (!moved && onSelect) {
            onSelect({ type: 'node', id: node.id });
          } else if (moved && onNodeMove) {
            onNodeMove(node.id, node.ui_position.x, node.ui_position.y, true);
          }
        });
      }

      svg.appendChild(g);
    });

    if (hooks.onCameraChange) hooks.onCameraChange(getScaleBarInfo());
  }

  return {
    render,
    fitToNodes,
    zoomBy,
    getCenterWorldPos,
    getScaleBarInfo,
  };
}
