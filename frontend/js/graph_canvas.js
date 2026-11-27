const ZONE_COLORS = {
  home: 'var(--graph-home)',
  campus: 'var(--graph-campus)',
  third_place: 'var(--graph-third-place)',
  commercial: 'var(--graph-third-place)',
  custom: 'var(--graph-custom)',
};

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

export function computePixelDistanceKm(posA, posB, pxPerKm = 120) {
  const dx = Number(posB?.x ?? 0) - Number(posA?.x ?? 0);
  const dy = Number(posB?.y ?? 0) - Number(posA?.y ?? 0);
  const km = Math.hypot(dx, dy) / Math.max(10, Number(pxPerKm || 120));
  return Math.max(0.1, Math.round(km * 10) / 10);
}

export function renderSpatialGraph(svg, worldConfig, options = {}) {
  const {
    interactive = true,
    selectedItem = null,
    connectSourceId = null,
    highlightedEdgeIds = [],
    highlightMode = 'fastest',
    agentNodeId = null,
    pxPerKm = 120,
    autoSyncDistance = true,
    onSelect = null,
    onNodeMove = null,
  } = options;

  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!worldConfig?.world_graph) return;

  const nodes = worldConfig.world_graph.locations || [];
  const edges = worldConfig.world_graph.edges || [];
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const highlighted = new Set(highlightedEdgeIds);

  const vb = svg.viewBox.baseVal;
  const maxX = (vb && vb.width ? vb.width : 820) - 48;
  const maxY = (vb && vb.height ? vb.height : 500) - 56;

  const xs = nodes.map((n) => Number(n.ui_position?.x ?? 0));
  const ys = nodes.map((n) => Number(n.ui_position?.y ?? 0));
  const maxAbs = Math.max(1, ...xs.map(Math.abs), ...ys.map(Math.abs));
  const toScreen = (pos) => {
    const rawX = Number(pos?.x ?? 0);
    const rawY = Number(pos?.y ?? 0);
    if (maxAbs <= 12) {
      return { x: 110 + rawX * 60, y: 80 + rawY * 55 };
    }
    return { x: Math.max(48, Math.min(maxX, rawX)), y: Math.max(42, Math.min(maxY, rawY)) };
  };

  // Track edge DOM refs so dragging a node updates connected edges smoothly at 60fps
  const edgeVisuals = [];
  const pairCounts = new Map();

  edges.forEach((edge) => {
    const src = nodeMap.get(edge.source);
    const dst = nodeMap.get(edge.target);
    if (!src || !dst) return;

    const key = [edge.source, edge.target].sort().join('::');
    const idx = pairCounts.get(key) || 0;
    pairCounts.set(key, idx + 1);
    const offset = idx === 0 ? 0 : (idx % 2 === 1 ? 1 : -1) * Math.ceil(idx / 2) * 22;

    const isSelected = selectedItem?.type === 'edge' && selectedItem.id === edge.id;
    const isHighlighted = highlighted.has(edge.id);

    const stroke = isHighlighted
      ? (highlightMode === 'cheapest' ? 'var(--graph-route-cheap)' : 'var(--graph-route-fast)')
      : (isSelected ? 'var(--ink-primary)' : 'var(--border-strong)');

    const path = svgEl('path', {
      fill: 'none',
      stroke,
      'stroke-width': isHighlighted || isSelected ? 3.2 : 1.75,
      'stroke-dasharray': edge.mode === 'walk' ? '6 4' : 'none',
      style: interactive ? 'cursor: pointer;' : '',
    });

    const hitPath = svgEl('path', {
      fill: 'none',
      stroke: 'transparent',
      'stroke-width': 16,
      style: interactive ? 'cursor: pointer;' : '',
    });

    const labelText = svgEl('text', {
      'text-anchor': 'middle',
      fill: isHighlighted ? stroke : 'var(--ink-secondary)',
      'font-size': 10,
      'font-family': 'var(--font-mono)',
      'font-weight': isHighlighted || isSelected ? 600 : 500,
      style: 'pointer-events: none;',
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
      const p1 = toScreen(src.ui_position);
      const p2 = toScreen(dst.ui_position);
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
      labelText.setAttribute('x', String(cx));
      labelText.setAttribute('y', String(cy - 6));
      const lockMark = edge._lockDistance ? ' [fixed]' : '';
      labelText.textContent = `${Number(edge.distance_km).toFixed(1)} km · ${edge.mode}${lockMark}`;
    });
  }

  updateEdgeGeometry();

  // Render nodes
  nodes.forEach((node) => {
    const pos = toScreen(node.ui_position);
    const isSelected = selectedItem?.type === 'node' && selectedItem.id === node.id;
    const isConnectSrc = connectSourceId === node.id;
    const isAgentHere = agentNodeId === node.id;
    const isHome = worldConfig.world_graph.home_node_id === node.id;
    const zoneColor = ZONE_COLORS[node.zone] || ZONE_COLORS.custom;

    const g = svgEl('g', {
      transform: `translate(${pos.x}, ${pos.y})`,
      style: interactive ? 'cursor: grab;' : '',
    });

    if (isAgentHere) {
      g.appendChild(svgEl('circle', {
        r: 28,
        fill: 'none',
        stroke: 'var(--graph-agent)',
        'stroke-width': 2.5,
      }));
    }

    if (isConnectSrc) {
      g.appendChild(svgEl('circle', {
        r: 27,
        fill: 'none',
        stroke: 'var(--ink-primary)',
        'stroke-width': 2,
        'stroke-dasharray': '4 3',
      }));
    }

    g.appendChild(svgEl('circle', {
      r: isSelected ? 22 : 20,
      fill: 'var(--bg-surface)',
      stroke: zoneColor,
      'stroke-width': isSelected ? 3.5 : 2.25,
    }));

    g.appendChild(svgEl('circle', {
      r: 7,
      fill: zoneColor,
    }));

    const title = svgEl('text', {
      y: 34,
      'text-anchor': 'middle',
      fill: 'var(--ink-primary)',
      'font-size': 11,
      'font-weight': 600,
      'font-family': 'var(--font-sans)',
    });
    const displayLabel = node.label || node.id;
    title.textContent = isHome ? `${displayLabel} (Home)` : displayLabel;
    g.appendChild(title);

    const caps = [
      node.can_study ? 'study' : null,
      node.has_kitchen ? 'kitchen' : null,
      node.is_class_venue ? 'class' : null,
      node.zone === 'third_place' ? 'cafe' : null,
    ].filter(Boolean).join(' · ');
    if (caps) {
      const sub = svgEl('text', {
        y: 46,
        'text-anchor': 'middle',
        fill: 'var(--ink-muted)',
        'font-size': 9.5,
        'font-family': 'var(--font-mono)',
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
        dragging = true;
        moved = false;
        startClientX = ev.clientX;
        startClientY = ev.clientY;
        g.setPointerCapture(ev.pointerId);
      });

      g.addEventListener('pointermove', (ev) => {
        if (!dragging) return;
        if (!moved && Math.hypot(ev.clientX - startClientX, ev.clientY - startClientY) < 4) {
          return;
        }
        moved = true;
        const rect = svg.getBoundingClientRect();
        const sx = ((ev.clientX - rect.left) / rect.width) * (vb.width || 820);
        const sy = ((ev.clientY - rect.top) / rect.height) * (vb.height || 500);
        const clampedX = Math.round(Math.max(48, Math.min(maxX, sx)));
        const clampedY = Math.round(Math.max(42, Math.min(maxY, sy)));
        node.ui_position = { x: clampedX, y: clampedY };
        g.setAttribute('transform', `translate(${clampedX}, ${clampedY})`);

        if (autoSyncDistance) {
          edges.forEach((edge) => {
            if (edge._lockDistance) return;
            if (edge.source === node.id || edge.target === node.id) {
              const other = nodeMap.get(edge.source === node.id ? edge.target : edge.source);
              if (other) {
                edge.distance_km = computePixelDistanceKm(node.ui_position, other.ui_position, pxPerKm);
              }
            }
          });
        }
        updateEdgeGeometry();
        if (onNodeMove) onNodeMove(node.id, clampedX, clampedY, false);
      });

      g.addEventListener('pointerup', (ev) => {
        if (!dragging) return;
        dragging = false;
        g.releasePointerCapture(ev.pointerId);
        if (!moved && onSelect) {
          onSelect({ type: 'node', id: node.id });
        } else if (moved && onNodeMove) {
          onNodeMove(node.id, node.ui_position.x, node.ui_position.y, true);
        }
      });
    }

    svg.appendChild(g);
  });
}
