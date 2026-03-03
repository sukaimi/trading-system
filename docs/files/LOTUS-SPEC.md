# LOTUS CREATURE — Claude Code Integration Spec
> Version 1.0 | Drop-in animated creature for the Trading System Dashboard

---

## What You're Integrating

`lotus-creature.js` is a self-contained canvas creature engine.
It connects to the existing WebSocket feed and REST API, reads trading events, and animates a living axolotl creature across 6 evolution stages in real-time.

**It requires zero new backend changes.**

---

## Files

```
lotus-creature/
└── lotus-creature.js    ← the entire engine, one file
```

---

## Step 1 — Add the canvas to your HTML

Find where you want the creature panel to live in your existing dashboard HTML.
Add a canvas element with an ID:

```html
<canvas id="lotus-canvas" width="400" height="500"></canvas>
```

Recommended: place it in a dedicated panel or sidebar column.
The canvas is fully self-contained — it renders its own background.

---

## Step 2 — Load the script

In your HTML `<head>` or before `</body>`:

```html
<script src="lotus-creature.js"></script>
```

If your project uses ES modules:
```js
import { LotusCreature } from './lotus-creature.js';
```

---

## Step 3 — Instantiate and init

```js
const lotus = new LotusCreature('lotus-canvas', {
  wsUrl:   'ws://localhost:8080/ws',   // your existing WS endpoint
  apiBase: '',                          // prefix if API is on different host
  width:   400,
  height:  500,
});

lotus.init(); // connects WS + loads initial portfolio/costs data
```

`init()` does two things automatically:
1. Fetches `/api/portfolio` and `/api/costs` to set initial state
2. Connects WebSocket at `wsUrl` and listens for events

---

## Step 4 — (Optional) Feed events manually

If you already have a WebSocket connection in your dashboard, you can
**share it** with the creature instead of opening a second connection:

```js
// Your existing WS handler
yourExistingSocket.onmessage = (msg) => {
  const event = JSON.parse(msg.data);

  // Your existing handling...
  handleDashboardEvent(event);

  // Also forward to creature
  lotus.dispatch(event);
};

// Then init without auto-connecting WS:
const lotus = new LotusCreature('lotus-canvas', {
  wsUrl: null,  // pass null to skip auto-connect
  apiBase: '',
});
lotus.init();
```

---

## Step 5 — (Optional) Sync portfolio manually

If you poll `/api/portfolio` on a timer (e.g. every 5s), you can push
updates directly to keep the creature in sync:

```js
// Inside your existing portfolio refresh loop:
const portfolio = await fetch('/api/portfolio').then(r => r.json());
updateDashboardUI(portfolio);    // your existing code
lotus.setPortfolio(portfolio);   // creature sync
```

---

## Step 6 — (Optional) Read creature state for your HUD

The creature exposes its current state so your dashboard can show
a stats panel (stage name, level, XP, mood, HP):

```js
// Call anytime — e.g. in your dashboard refresh loop
const state = lotus.getState();

console.log(state);
// {
//   stage:      2,
//   stageName: 'Lotus',
//   level:      7,
//   xp:         450,
//   xpToNext:  1000,
//   mood:      'battling',
//   regression: false,
//   sleeping:   false,
//   winStreak:  3,
//   hp:         78,
//   palette:    { body, glow, eye, accent, bg, particle }
// }
```

Use `state.palette.glow` to tint your HUD elements to match the creature's current stage.

---

## Expected WebSocket Event Format

The creature listens for events in this shape (matching your existing pipeline):

```json
{
  "category": "pipeline",
  "event_type": "trade_executed",
  "data": {
    "direction": "BUY",
    "symbol": "BTCUSDT",
    "confidence": 0.82
  }
}
```

### Categories the creature responds to:

| category | event_types listened |
|---|---|
| `pipeline` | `news_scan_start`, `news_scan_complete`, `thesis_generated`, `no_thesis`, `devil_verdict`, `risk_check`, `trade_executed`, `trade_killed` |
| `portfolio` | `updated` |
| `cost` | any (reads `cost_usd`, `provider`) |
| `heartbeat` | any |
| `scheduler` | `task_run` |
| `circuit_breaker` | `triggered`, `reset` |

Events with unrecognised categories are silently ignored — safe to use alongside any existing listeners.

---

## Evolution Trigger Reference

| Stage | Name | Condition |
|---|---|---|
| 0 | Dormant | No trades yet |
| 1 | Larva | First trade executed |
| 2 | Lotus | 10+ total trades |
| 3 | Elder | 50+ total trades |
| 4 | Storm | 100+ trades AND active win streak ≥5 |
| 5 | Apex | 200+ total trades |

Storm de-activates (reverts to Elder) when win streak drops below 5.
Regression (greyscale sick state) triggers on `circuit_breaker.triggered` and clears on `circuit_breaker.reset`.

---

## XP System

| Action | XP Gained |
|---|---|
| Trade executed | +20 |
| Thesis generated | +3 |
| News scan | +2 |
| Trade killed | +5 |

Level formula: `xpToNext = floor(100 × 1.4^level)`

---

## CSS Sizing Note

The canvas renders at the dimensions passed in the constructor (`width`/`height`).
For responsive sizing, control the canvas with CSS — the internal resolution stays fixed:

```css
#lotus-canvas {
  width: 100%;
  max-width: 400px;
  height: auto;
}
```

---

## Troubleshooting

**Creature is not animating:**
- Check browser console for errors
- Confirm canvas ID matches what was passed to constructor

**WebSocket not connecting:**
- Confirm `wsUrl` matches your actual WS endpoint
- If using `wss://` (HTTPS), make sure server supports it

**Creature stuck on Dormant:**
- Check `/api/portfolio` returns `total_trades > 0`
- Or dispatch a `trade_executed` event manually to test

**All stages look the same colour:**
- The palette changes on evolution — trigger more trades to advance stages
- Or temporarily lower `STAGE_THRESHOLDS` in the class for testing

---

## Testing Stages Without Live Data

Paste in browser console to jump to any stage:

```js
// Jump to stage 3 (Elder)
lotus.stage = 3;
lotus.palette = lotus._getPalette(3);
lotus._particles = lotus._makeParticles();

// Trigger a trade effect
lotus.dispatch({ category: 'pipeline', event_type: 'trade_executed', data: { direction: 'BUY' } });

// Trigger circuit breaker regression
lotus.dispatch({ category: 'circuit_breaker', event_type: 'triggered', data: {} });

// Recover from regression
lotus.dispatch({ category: 'circuit_breaker', event_type: 'reset', data: {} });
```
