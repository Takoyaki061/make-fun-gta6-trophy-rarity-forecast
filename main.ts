// gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI PS5 trophy's Exophase
// earn rate at the case target, as five quantiles on the logit scale.
//
// Model (own design, fitted from the case file only, no network):
//   r = E / P, earners over players tracked. Recover E = r * P / 100 per read.
//   Two regimes, chosen by whether the history actually moves:
//   * RICH history (several distinct reads): blend of two per-trophy models.
//     B, saturating growth r(t) = R * (1 - exp(-t/T)), fitted by grid search
//     over T with R by least squares: a trophy's rate approaches its ceiling
//     instead of growing forever. C, the log-time slope of logit(r) over the
//     LAST 8 distinct reads only, shrunk toward the cross-trophy median and
//     anchored on the last read: the recent slope carries the current growth
//     rate, not the early steep part. The median is the average of the two
//     models' logits, and the spread adds their disagreement.
//   * FLAT history (one distinct read, e.g. launch night before the first
//     refresh): the page is stale, so the answer is most likely a refresh:
//     P jumps and rates dilute. Shift the median down by a dilution amount
//     that grows as the trophy gets rarer (new players almost never earn the
//     rare ones quickly), except the 100% trophy, which stays 100%.
//   Spread: a logit sigma per trophy from a refresh-shock floor, the fit
//   residual (rich path) and a random-walk term in horizon, so the 5%-95%
//   band is honest about the common refresh shock instead of the house
//   baseline's narrow 0.5*sqrt(days) band that the practice refresh blows
//   through. Quantiles are the median's logit plus z_tau * sigma, mapped
//   back through the logistic, rounded to 4 decimals.
//
// Deterministic: no randomness, no clock reads, same bytes every run.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Hours from the case's first read to a timestamp (log-time clock). If a
 * timestamp does not parse, 0: the fits then see no time passing and fall
 * back to a near-persistence forecast instead of producing NaNs. */
function hoursSinceStart(at: string, startMs: number): number {
  const ms = Date.parse(at);
  if (Number.isNaN(ms) || Number.isNaN(startMs)) return 0;
  return Math.max(0, (ms - startMs) / 3_600_000);
}

/**
 * Dilution shift (logit units) for a flat history: how far a trophy's rate
 * falls at the refresh that the stale page is hiding. Rarer trophies fall
 * further, because a fresh cohort of new players earns them almost never
 * in the first hours, while the 100% trophy is earned by everyone at once.
 * Calibrated on the practice refresh (P 638 -> 3,868, every rate but the
 * 100% one lower): a typical trophy's rate roughly halves, the rarest fall
 * to about a quarter.
 */
function dilutionShift(rate: number): number {
  if (rate >= 100) return 0;
  if (rate >= 1.5) return Math.log(0.65);
  if (rate >= 1.0) return Math.log(0.5);
  if (rate >= 0.8) return Math.log(0.35);
  return Math.log(0.25);
}

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const n = c.trophies.length;
  const horizonDays = c.horizon_minutes / 1440;
  const startMs = Date.parse(c.history[0].at);
  const targetMs = Date.parse(c.target);
  const tTargetRaw = (targetMs - startMs) / 3_600_000;
  const tTarget = Number.isNaN(tTargetRaw) ? 0 : Math.max(0, tTargetRaw);

  // Collapse the history to its distinct reads (consecutive identical rate
  // vectors are one Exophase refresh, not new information).
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (prev && JSON.stringify(prev.rates) === JSON.stringify(r.rates)
      && prev.players_tracked === r.players_tracked) continue;
    distinct.push(r);
  }
  const last = c.history[c.history.length - 1];
  const flat = distinct.length <= 1;

  // Per-trophy rich-history fit: two models blended, because no single shape
  // is right for every trophy and the true progression curve is unknown.
  //   B: saturating growth r(t) = R * (1 - exp(-t/T)), fitted by a grid
  //      search over T (R solved by least squares per T). It cannot run
  //      away: a trophy's rate approaches its ceiling R instead of growing
  //      forever, which is what progression does.
  //   C: log-time slope of logit(r) over the LAST 8 distinct reads only,
  //      shrunk toward the cross-trophy median, anchored on the last read.
  //      The recent slope carries the current growth rate, not the early
  //      steep part that a whole-history fit extrapolates into overshoot.
  // The median is the average of the two models' logits. (A whole-history
  // log-time slope was tried first and rejected: on synthetic rich cases it
  // overshot badly — typical miss ~16 points vs ~1-3 for this blend —
  // because it extrapolates the early steep rise past the saturation.)
  const TG = [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536];
  const medB: number[] = new Array(n).fill(0);
  const medC: number[] = new Array(n).fill(0);
  const residuals: number[] = new Array(n).fill(0.3);
  if (!flat) {
    const xs = distinct.map((r) => Math.log1p(hoursSinceStart(r.at, startMs)));
    const tHours = distinct.map((r) => hoursSinceStart(r.at, startMs));
    for (let i = 0; i < n; i++) {
      // --- Model B: exponential saturation ---
      let best: { ss: number; R: number; T: number } | null = null;
      for (const T of TG) {
        let num = 0;
        let den = 0;
        for (let j = 0; j < distinct.length; j++) {
          const r = distinct[j].rates[i];
          if (r === null) continue;
          const g = 1 - Math.exp(-tHours[j] / T);
          num += r * g;
          den += g * g;
        }
        if (den === 0) continue;
        const R = Math.min(100, Math.max(0.01, num / den));
        let ss = 0;
        for (let j = 0; j < distinct.length; j++) {
          const r = distinct[j].rates[i];
          if (r === null) continue;
          const g = 1 - Math.exp(-tHours[j] / T);
          ss += (r - R * g) ** 2;
        }
        if (best === null || ss < best.ss) best = { ss, R, T };
      }
      const Rb = best === null ? 0.01 : best.R;
      const Tb = best === null ? 48 : best.T;
      medB[i] = logit(Math.min(99.99, Math.max(0.01, Rb * (1 - Math.exp(-tTarget / Tb)))));
      // --- Model C: recent-window log-time slope ---
      const K = Math.min(8, distinct.length);
      const rec = distinct.slice(distinct.length - K);
      const recXs = xs.slice(xs.length - K);
      const ys: number[] = [];
      const xx: number[] = [];
      for (let j = 0; j < rec.length; j++) {
        const r = rec[j].rates[i];
        if (r === null) continue;
        ys.push(logit(r));
        xx.push(recXs[j]);
      }
      if (ys.length >= 3) {
        const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;
        const xMean = xx.reduce((a, b) => a + b, 0) / xx.length;
        let num = 0;
        let den = 0;
        for (let k = 0; k < xx.length; k++) {
          num += (xx[k] - xMean) * (ys[k] - yMean);
          den += (xx[k] - xMean) ** 2;
        }
        const b = den > 0 ? num / den : 0;
        let ss = 0;
        for (let k = 0; k < xx.length; k++) {
          const fit = yMean + b * (xx[k] - xMean);
          ss += (ys[k] - fit) ** 2;
        }
        residuals[i] = Math.sqrt(ss / Math.max(1, xx.length - 2));
        medC[i] = b; // slope stored temporarily; converted below
      } else {
        medC[i] = 0;
        residuals[i] = 0.3;
      }
    }
    // Shrink the recent slopes toward the cross-trophy median, then convert
    // them to a target logit anchored on the last read.
    const slopes = medC.slice();
    const med = median(slopes);
    const w = Math.min(8, distinct.length) / (Math.min(8, distinct.length) + 5);
    const tLast = hoursSinceStart(last.at, startMs);
    const dLogT = Math.log1p(tTarget) - Math.log1p(tLast);
    for (let i = 0; i < n; i++) {
      const b = w * slopes[i] + (1 - w) * med;
      const rLast = last.rates[i];
      const base = rLast === null || rLast <= 0 ? 0.01 : rLast;
      const move = Math.max(-4, Math.min(4, b * dLogT));
      medC[i] = logit(base) + move;
    }
  }

  return c.trophies.map((t, i) => {
    const rLast = last.rates[i];
    // The 100% trophy is earned by everyone the moment they start: it stays.
    if (rLast !== null && rLast >= 100) {
      return { id: t.id, q: [100, 100, 100, 100, 100] };
    }
    const base = rLast === null || rLast <= 0 ? 0.01 : rLast;
    let medLogit: number;
    let sigma: number;
    if (flat) {
      // Stale page: the answer is a refresh, rates dilute.
      medLogit = logit(base) + dilutionShift(base);
      // Wide and honest: the refresh size is the dominant unknown, and it
      // grows with the horizon (more refreshes, more dilution possible).
      sigma = Math.sqrt(1.0 ** 2 + (0.5 * Math.sqrt(horizonDays)) ** 2);
    } else {
      // Rich history: blend of the saturating fit (B) and the recent slope
      // (C). The spread adds the two models' disagreement, so a trophy
      // the models disagree on gets a wider band.
      const medLogitB = medB[i];
      const medLogitC = medC[i];
      medLogit = (medLogitB + medLogitC) / 2;
      const disagree = Math.abs(medLogitB - medLogitC);
      const resid = Math.max(0.2, Math.min(1.0, residuals[i]));
      sigma = Math.sqrt(0.5 ** 2 + resid ** 2 + (0.5 * disagree) ** 2 + (0.5 * Math.sqrt(horizonDays)) ** 2);
    }
    const q = Z.map((z) => round4(Math.min(100, Math.max(0, sigmoidPct(medLogit + z * sigma)))));
    // Keep the row non-decreasing after rounding (rounding can tie, never invert).
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history.length) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
