export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function formatIdr(amount) {
  return `Rp ${Math.round(Number(amount || 0)).toLocaleString('id-ID')}`;
}

const MONTHS_ID = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];

export function formatRunDisplayLabel(run) {
  if (!run) return '-';
  const name = run.experiment_name || run.run_id || 'model';
  let dateLabel = '';
  const ts = String(run.timestamp || '');
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  if (m) {
    const day = Number(m[3]);
    const mon = MONTHS_ID[Math.max(0, Math.min(11, Number(m[2]) - 1))];
    dateLabel = `${day} ${mon} ${m[4]}:${m[5]}`;
  }
  const s = run.final_summary;
  let scoreLabel = '';
  if (s && Number.isFinite(Number(s.completion_rate))) {
    const pct = Math.round(Number(s.completion_rate) * 100);
    const bal = Math.round(Number(s.mean_final_balance_idr || 0) / 1000);
    scoreLabel = ` · ${pct}% · ${bal.toLocaleString('id-ID')}rb`;
  }
  return dateLabel ? `${name} · ${dateLabel}${scoreLabel}` : `${name}${scoreLabel}`;
}

export function nextUniqueId(prefix, existingIds) {
  const taken = new Set(existingIds);
  let i = 1;
  while (taken.has(`${prefix}_${i}`)) i += 1;
  return `${prefix}_${i}`;
}

/**
 * Menjaga referensi activities[*].target_nodes selalu valid dan sinkron
 * dengan kapabilitas lokasi serta home_node_id terbaru di world_graph.
 */
export function syncActivitiesWithGraph(worldConfig) {
  const wg = worldConfig?.world_graph;
  const acts = worldConfig?.activities;
  if (!wg || !Array.isArray(acts)) return;

  const nodes = wg.locations || [];
  const validIds = new Set(nodes.map((n) => n.id));
  const homeId = wg.home_node_id;

  const classNodes = nodes.filter((n) => n.is_class_venue).map((n) => n.id);
  const sleepNodes = nodes.filter((n) => n.can_sleep).map((n) => n.id);
  const groceryNodes = nodes.filter((n) => n.has_grocery).map((n) => n.id);
  const hubNodes = nodes
    .filter((n) => n.can_study && (n.study_profile?.craving_relief || 0) > 0 && !n.study_profile?.craving_reset)
    .map((n) => n.id);
  const leisureNodes = nodes
    .filter((n) => n.zone === 'third_place' || n.study_profile?.craving_reset)
    .map((n) => n.id);
  const homeStudyNodes = nodes.filter((n) => n.can_sleep && n.can_study).map((n) => n.id);
  const waitNodes = nodes.filter((n) => !n.can_sleep && n.closes_at_night).map((n) => n.id);

  acts.forEach((act) => {
    if (Array.isArray(act.target_nodes)) {
      act.target_nodes = act.target_nodes.filter((id) => validIds.has(id));
    }
    switch (act.handler) {
      case 'rest':
        act.target_nodes = sleepNodes.includes(homeId) ? [homeId] : (sleepNodes.length ? sleepNodes : null);
        break;
      case 'grocery_shop':
        act.target_nodes = groceryNodes.length ? groceryNodes : null;
        break;
      case 'night_cram':
        act.target_nodes = homeStudyNodes.includes(homeId) ? [homeId] : (homeStudyNodes.length ? homeStudyNodes : null);
        break;
      case 'attend_schedule':
        act.target_nodes = classNodes.length ? classNodes : null;
        break;
      case 'study':
        act.target_nodes = homeStudyNodes.length ? homeStudyNodes : null;
        break;
      case 'hub_stay':
        act.target_nodes = hubNodes.length ? hubNodes : null;
        break;
      case 'leisure_reset':
        act.target_nodes = leisureNodes.length ? leisureNodes : null;
        break;
      case 'wait':
        act.target_nodes = waitNodes.length ? waitNodes : null;
        break;
      default:
        if (Array.isArray(act.target_nodes) && act.target_nodes.length === 0) {
          act.target_nodes = null;
        }
        break;
    }
  });
}
