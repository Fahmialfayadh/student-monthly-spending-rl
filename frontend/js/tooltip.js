/* Viewport-clamped floating tooltip controller (hover-only, gentle 360ms delay so it never intrudes on typing) */

let tooltipEl = null;
let showTimer = null;
let activeTarget = null;

function ensureTooltipEl() {
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'floating-tooltip';
    tooltipEl.setAttribute('role', 'tooltip');
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}

function positionTooltip(target) {
  const el = ensureTooltipEl();
  const text = target.getAttribute('data-tooltip');
  if (!text) return;
  el.textContent = text;
  el.style.left = '-9999px';
  el.style.top = '-9999px';
  el.classList.add('is-visible');

  const rect = target.getBoundingClientRect();
  const tipRect = el.getBoundingClientRect();
  const pad = 10;

  let left = rect.left + rect.width / 2 - tipRect.width / 2;
  left = Math.max(pad, Math.min(window.innerWidth - tipRect.width - pad, left));

  let top = rect.top - tipRect.height - 8;
  if (top < pad) {
    top = rect.bottom + 8;
  }

  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;
}

export function hideTooltip() {
  clearTimeout(showTimer);
  activeTarget = null;
  if (tooltipEl) tooltipEl.classList.remove('is-visible');
}

export function initGlobalTooltips() {
  document.addEventListener('mouseover', (ev) => {
    const target = ev.target?.closest?.('[data-tooltip]');
    if (!target) {
      if (activeTarget) hideTooltip();
      return;
    }
    if (target === activeTarget) return;
    hideTooltip();
    activeTarget = target;
    showTimer = setTimeout(() => {
      if (activeTarget === target) positionTooltip(target);
    }, 360);
  });

  document.addEventListener('mouseout', (ev) => {
    if (!activeTarget) return;
    const related = ev.relatedTarget?.closest?.('[data-tooltip]');
    if (related !== activeTarget) hideTooltip();
  });

  // Hide immediately on click or input focus so tooltip never blocks typing
  document.addEventListener('pointerdown', hideTooltip);
  document.addEventListener('focusin', hideTooltip);
  window.addEventListener('scroll', hideTooltip, true);
}
