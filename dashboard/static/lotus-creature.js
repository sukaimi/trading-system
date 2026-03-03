/**
 * ═══════════════════════════════════════════════════════════════
 * LOTUS CREATURE ENGINE v1.0
 * ═══════════════════════════════════════════════════════════════
 *
 * Self-contained animated creature for the Trading System Dashboard.
 * Connects to the existing WebSocket feed and API endpoints.
 *
 * USAGE:
 *   const lotus = new LotusCreature('canvas-id', { wsUrl: 'ws://localhost:8080/ws' });
 *   lotus.init();
 *
 * The creature reads portfolio/pipeline/cost/heartbeat events
 * and reacts visually in real-time across 6 evolution stages.
 * ═══════════════════════════════════════════════════════════════
 */

class LotusCreature {

  // ─────────────────────────────────────────
  // CONSTRUCTOR
  // ─────────────────────────────────────────
  constructor(canvasId, options = {}) {
    this.canvas = typeof canvasId === 'string'
      ? document.getElementById(canvasId)
      : canvasId;
    this.ctx = this.canvas.getContext('2d');

    this.opts = Object.assign({
      wsUrl:    `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
      apiBase:  '',
      width:    this.canvas.width  || 400,
      height:   this.canvas.height || 500,
    }, options);

    this.canvas.width  = this.opts.width;
    this.canvas.height = this.opts.height;

    // ── Core state ──────────────────────────
    this.t = 0;           // animation clock
    this.stage = 0;       // 0–5 evolution stage
    this.regression = false;

    // ── Portfolio / trading data ─────────────
    this.data = {
      equity:       0,
      dailyPnl:     0,
      drawdown:     0,
      wins:         0,
      losses:       0,
      totalTrades:  0,
      openPositions: [],
      halted:       false,
      winStreak:    0,
      llmCosts:     { total: 0, byProvider: {} },
    };

    // ── Creature stats ───────────────────────
    this.xp          = 0;
    this.xpToNext    = 100;
    this.level       = 1;
    this.mood        = 'idle';   // idle | happy | sad | excited | battling | sick | sleeping

    // ── Animation effect queue ───────────────
    this.effects     = [];       // [{type, duration, elapsed, data}]
    this.idleTimer   = 0;        // seconds since last event
    this.sleeping    = false;

    // ── Egg crack system ─────────────────────
    this.cracks      = [];       // [{x,y,angle,len,branches}]
    this.eggGlow     = 0;        // 0–1

    // ── Cached sub-state ─────────────────────
    this._particles  = this._makeParticles();
    this._orbitPts   = this._makeOrbitPoints();
    this._stars      = this._makeStars();

    // ── Palette for current stage ─────────────
    this.palette     = this._getPalette(0);
  }

  // ─────────────────────────────────────────
  // PUBLIC: init — loads data, starts loop, connects WS
  // ─────────────────────────────────────────
  async init() {
    await this._loadInitialData();
    this._connectWS();
    this._loop();
  }

  // ─────────────────────────────────────────
  // PUBLIC: dispatch — call this if you already have a WS
  //   feed and want to feed events manually
  // ─────────────────────────────────────────
  dispatch(event) {
    this._handleEvent(event);
  }

  // ─────────────────────────────────────────
  // PUBLIC: setPortfolio — sync portfolio state from outside
  // ─────────────────────────────────────────
  setPortfolio(p) {
    this._applyPortfolio(p);
  }

  // ═════════════════════════════════════════
  // INTERNAL: DATA LOADING
  // ═════════════════════════════════════════

  async _loadInitialData() {
    try {
      const [portfolio, costs] = await Promise.all([
        fetch(`${this.opts.apiBase}/api/portfolio`).then(r => r.ok ? r.json() : {}),
        fetch(`${this.opts.apiBase}/api/costs`).then(r => r.ok ? r.json() : {}),
      ]);
      if (portfolio) this._applyPortfolio(portfolio);
      if (costs)     this._applyCosts(costs);
    } catch (e) { /* silent — creature still works offline */ }
  }

  _applyPortfolio(p) {
    const prev = { ...this.data };
    this.data.equity        = p.equity || 0;
    this.data.dailyPnl      = p.daily_pnl_pct || 0;
    this.data.drawdown      = p.drawdown_from_peak_pct || 0;
    this.data.wins          = p.total_wins || 0;
    this.data.losses        = p.total_losses || 0;
    this.data.totalTrades   = p.total_trades || 0;
    this.data.openPositions = p.open_positions || [];
    this.data.halted        = p.halted || false;

    // Derive win streak from delta (best-effort)
    if (this.data.wins > prev.wins)   this.data.winStreak++;
    if (this.data.losses > prev.losses) this.data.winStreak = 0;

    this._syncStageToData();
    this._syncMoodToData();
  }

  _applyCosts(c) {
    this.data.llmCosts.total      = c.total_usd || 0;
    this.data.llmCosts.byProvider = c.by_provider || {};
  }

  // ═════════════════════════════════════════
  // INTERNAL: WEBSOCKET
  // ═════════════════════════════════════════

  _connectWS() {
    if (!this.opts.wsUrl) return;
    const ws = new WebSocket(this.opts.wsUrl);
    ws.onmessage = (msg) => {
      try { this._handleEvent(JSON.parse(msg.data)); } catch(e) {}
    };
    ws.onclose = () => setTimeout(() => this._connectWS(), 3000);
  }

  _handleEvent(ev) {
    this.idleTimer = 0;
    this.sleeping  = false;

    const d = ev.data || {};

    switch (ev.category) {

      // ── Portfolio updates ──────────────────
      case 'portfolio':
        if (ev.event_type === 'updated' && d) this._applyPortfolio(d);
        break;

      // ── Pipeline events ────────────────────
      case 'pipeline':
        this._onPipelineEvent(ev.event_type, d);
        break;

      // ── Cost / LLM ────────────────────────
      case 'cost':
        if (d.cost_usd) {
          this.data.llmCosts.total += d.cost_usd;
          const prov = d.provider || 'unknown';
          this.data.llmCosts.byProvider[prov] =
            (this.data.llmCosts.byProvider[prov] || 0) + d.cost_usd;
          this._onLlmCall(prov, d.cost_usd);
        }
        break;

      // ── Heartbeat ─────────────────────────
      case 'heartbeat':
        this._onHeartbeat();
        break;

      // ── Scheduler ─────────────────────────
      case 'scheduler':
        if (ev.event_type === 'task_run') this._onSchedulerTask();
        break;

      // ── Circuit breaker ───────────────────
      case 'circuit_breaker':
        this._onCircuitBreaker(ev.event_type, d);
        break;
    }
  }

  // ═════════════════════════════════════════
  // INTERNAL: PIPELINE EVENT HANDLER
  // ═════════════════════════════════════════

  _onPipelineEvent(type, d) {
    switch (type) {

      case 'news_scan_start':
        this._queueEffect('news_scan');
        if (this.stage === 0) this.eggGlow = Math.min(1, this.eggGlow + 0.15);
        break;

      case 'news_scan_complete':
        if (this.stage === 0) this._queueEffect('egg_pulse');
        break;

      case 'thesis_generated':
        this._queueEffect('thesis', { confidence: d.confidence || 0 });
        if (this.stage === 0) this._addEggCrack();
        this._gainXP(3);
        break;

      case 'no_thesis':
        this._queueEffect('look_around');
        break;

      case 'devil_verdict':
        if (d.verdict === 'APPROVED') {
          this._queueEffect('devil_approved');
          if (this.stage === 0) this._addEggCrack();
        } else if (d.verdict === 'APPROVED_WITH_MODIFICATION') {
          this._queueEffect('devil_modified');
        } else if (d.verdict === 'KILLED') {
          this._queueEffect('devil_killed');
          if (this.stage === 0) {
            // crack seals slightly
            this.eggGlow = Math.max(0, this.eggGlow - 0.1);
          }
        }
        break;

      case 'risk_check':
        if (d.approved) this._queueEffect('risk_approved');
        else            this._queueEffect('risk_rejected');
        break;

      case 'trade_executed':
        this._onTradeExecuted(d);
        break;

      case 'trade_killed':
        this._queueEffect('trade_killed');
        break;
    }
  }

  _onTradeExecuted(d) {
    this._gainXP(20);
    this.data.totalTrades++;

    if (this.stage === 0) {
      // HATCH SEQUENCE
      this._queueEffect('hatch', { duration: 2000 });
      setTimeout(() => {
        this.stage = 1;
        this.palette = this._getPalette(1);
        this._particles = this._makeParticles();
      }, 2000);
    } else {
      this._queueEffect('trade_executed', { direction: d.direction });
    }
  }

  _onLlmCall(provider, cost) {
    // Each provider maps to a bubble colour — visible in Lotus+
    this._queueEffect('llm_bubble', { provider, cost });
    if (this.stage === 0) {
      this.eggGlow = Math.min(1, this.eggGlow + 0.05);
    }
  }

  _onHeartbeat() {
    this._queueEffect('heartbeat_breath');
  }

  _onSchedulerTask() {
    this._queueEffect('gill_perk');
  }

  _onCircuitBreaker(type, d) {
    if (type === 'triggered') {
      this.regression = true;
      this._queueEffect('circuit_breaker');
      // De-evolve one stage visually (not permanently)
      const regStage = Math.max(0, this.stage - 1);
      this.palette = this._getPalette(regStage, true); // true = greyscale
    } else if (type === 'reset' || d?.decision === 'RESUME') {
      this.regression = false;
      this.palette = this._getPalette(this.stage);
      this._queueEffect('recovery');
    }
  }

  // ═════════════════════════════════════════
  // INTERNAL: EVOLUTION / XP SYSTEM
  // ═════════════════════════════════════════

  // Stage thresholds (total trades)
  static STAGE_THRESHOLDS = [0, 1, 10, 50, 100, 200];

  // XP thresholds per level
  static XP_CURVE = (level) => Math.floor(100 * Math.pow(1.4, level));

  _syncStageToData() {
    if (this.regression) return;
    const trades = this.data.totalTrades;
    let newStage = 0;
    const t = LotusCreature.STAGE_THRESHOLDS;
    for (let i = t.length - 1; i >= 0; i--) {
      if (trades >= t[i]) { newStage = i; break; }
    }
    // Storm (stage 4) also requires active win streak
    if (newStage >= 4 && this.data.winStreak < 5) newStage = 3;

    if (newStage !== this.stage && newStage > this.stage) {
      this._evolve(newStage);
    }
  }

  _evolve(newStage) {
    this.stage   = newStage;
    this.palette = this._getPalette(newStage);
    this._particles = this._makeParticles();
    this._queueEffect('evolve', { stage: newStage, duration: 2500 });
  }

  _gainXP(amount) {
    this.xp += amount;
    while (this.xp >= this.xpToNext) {
      this.xp      -= this.xpToNext;
      this.level++;
      this.xpToNext = LotusCreature.XP_CURVE(this.level);
      this._queueEffect('level_up');
    }
  }

  _syncMoodToData() {
    const { dailyPnl, drawdown, openPositions, halted, winStreak } = this.data;
    if (halted)                    this.mood = 'sick';
    else if (drawdown > 10)        this.mood = 'sad';
    else if (winStreak >= 5)       this.mood = 'excited';
    else if (dailyPnl > 2)         this.mood = 'happy';
    else if (dailyPnl < -1)        this.mood = 'sad';
    else if (openPositions.length) this.mood = 'battling';
    else                           this.mood = 'idle';
  }

  // ═════════════════════════════════════════
  // INTERNAL: EFFECT QUEUE
  // ═════════════════════════════════════════

  _queueEffect(type, data = {}) {
    const durations = {
      trade_executed:    800,
      evolve:           2500,
      hatch:            2000,
      level_up:         1200,
      circuit_breaker:  1500,
      recovery:          800,
      thesis:            600,
      devil_killed:      500,
      devil_approved:    400,
      devil_modified:    400,
      risk_approved:     300,
      risk_rejected:     400,
      news_scan:         400,
      egg_pulse:         600,
      look_around:       800,
      heartbeat_breath:  400,
      gill_perk:         500,
      llm_bubble:        600,
      trade_killed:      500,
      flash_win:         500,
      flash_loss:        500,
    };
    this.effects.push({
      type,
      duration: data.duration || durations[type] || 500,
      elapsed:  0,
      data,
    });
    // Cap queue length to avoid stacking
    if (this.effects.length > 6) this.effects.shift();
  }

  _tickEffects(dt) {
    this.effects = this.effects.filter(e => {
      e.elapsed += dt;
      return e.elapsed < e.duration;
    });
  }

  _hasEffect(type) {
    return this.effects.some(e => e.type === type);
  }

  _effectProgress(type) {
    const e = this.effects.find(e => e.type === type);
    return e ? e.elapsed / e.duration : 0;
  }

  // ═════════════════════════════════════════
  // INTERNAL: PALETTES
  // ═════════════════════════════════════════

  _getPalette(stage, grey = false) {
    const palettes = [
      // 0: Dormant
      { body: [170,155,210], glow: [200,190,240], eye: [180,170,230], accent: [210,200,255], bg: [13,10,24], particle: [170,155,210] },
      // 1: Larva
      { body: [120,190,255], glow: [180,220,255], eye: [160,210,255], accent: [200,235,255], bg: [14,21,32], particle: [120,190,255] },
      // 2: Lotus
      { body: [255,140,200], glow: [255,180,230], eye: [220,160,255], accent: [255,210,240], bg: [20,14,26], particle: [255,140,200] },
      // 3: Elder
      { body: [140,80,255],  glow: [200,170,255], eye: [210,190,255], accent: [220,200,255], bg: [14,10,32], particle: [140,80,255] },
      // 4: Storm
      { body: [100,220,255], glow: [220,245,255], eye: [240,250,255], accent: [255,255,220], bg: [8,13,24],  particle: [100,220,255] },
      // 5: Apex
      { body: [60,35,10],    glow: [255,200,60],  eye: [255,220,80],  accent: [255,180,40],  bg: [16,10,8],  particle: [255,200,60] },
    ];
    const p = { ...palettes[Math.min(stage, 5)] };
    if (grey) {
      // Desaturate everything
      const desat = (c) => {
        const avg = Math.round((c[0]+c[1]+c[2])/3);
        return [avg, avg, avg];
      };
      Object.keys(p).forEach(k => {
        if (Array.isArray(p[k])) p[k] = desat(p[k]);
      });
    }
    return p;
  }

  // ═════════════════════════════════════════
  // INTERNAL: PARTICLE SYSTEMS
  // ═════════════════════════════════════════

  _makeParticles() {
    return Array.from({ length: 35 }, () => ({
      x:     Math.random() * this.opts.width,
      y:     Math.random() * this.opts.height,
      vx:    (Math.random() - 0.5) * 0.4,
      vy:    -(Math.random() * 0.5 + 0.1),
      size:  Math.random() * 2 + 0.5,
      alpha: Math.random(),
      phase: Math.random() * Math.PI * 2,
    }));
  }

  _makeOrbitPoints() {
    return Array.from({ length: 18 }, (_, i) => ({
      angle:  (i / 18) * Math.PI * 2,
      r:      65 + Math.random() * 25,
      speed:  (0.008 + Math.random() * 0.012) * (Math.random() > 0.5 ? 1 : -1),
      size:   Math.random() * 2 + 1,
      phase:  Math.random() * Math.PI * 2,
    }));
  }

  _makeStars() {
    return Array.from({ length: 40 }, () => ({
      x:     Math.random() * this.opts.width,
      y:     Math.random() * this.opts.height,
      size:  Math.random() * 1.5 + 0.3,
      phase: Math.random() * Math.PI * 2,
    }));
  }

  // ═════════════════════════════════════════
  // INTERNAL: EGG CRACK SYSTEM
  // ═════════════════════════════════════════

  _addEggCrack() {
    const cx = this.opts.width / 2;
    const cy = this.opts.height * 0.44;
    const count = this.cracks.length;
    // Each crack starts from egg surface and branches
    const angle = (count * 1.1 + Math.random() * 0.4) % (Math.PI * 2);
    this.cracks.push({
      startAngle: angle,
      len:        10 + Math.random() * 20,
      branches:   [],
      glow:       1,
      // start point on egg surface
      ox: cx + Math.cos(angle) * 44,
      oy: cy + Math.sin(angle) * 50,
    });
    this.eggGlow = Math.min(1, this.eggGlow + 0.2);
  }

  // ═════════════════════════════════════════
  // INTERNAL: MAIN RENDER LOOP
  // ═════════════════════════════════════════

  _loop() {
    const dt = 16; // ~60fps fixed step
    this.t   += dt / 1000;
    this.idleTimer += dt / 1000;

    // Sleep after 5 min idle
    if (this.idleTimer > 300 && this.stage >= 2) {
      this.sleeping = true;
      this.mood = 'sleeping';
    }

    this._tickEffects(dt);
    this._render();
    requestAnimationFrame(() => this._loop());
  }

  _render() {
    const { ctx, opts } = this;
    const W = opts.width, H = opts.height;
    const cx = W / 2, cy = H * 0.48;

    ctx.clearRect(0, 0, W, H);

    // Background
    this._drawBackground(cx, cy);

    // Stage-specific environment
    if (this.stage >= 1) this._drawParticles(cx, cy);
    if (this.stage === 4) this._drawOrbitParticles(cx, cy);
    if (this.stage === 5) this._drawNebula(cx, cy);

    // Regression overlay
    if (this.regression) this._drawRegressionOverlay(cx, cy);

    // Creature
    switch (this.stage) {
      case 0: this._drawEgg(cx, cy);       break;
      case 1: this._drawLarva(cx, cy);     break;
      case 2: this._drawLotus(cx, cy);     break;
      case 3: this._drawElder(cx, cy);     break;
      case 4: this._drawStorm(cx, cy);     break;
      case 5: this._drawApex(cx, cy);      break;
    }

    // Effect overlays (flashes etc.)
    this._drawEffectOverlays(cx, cy);
  }

  // ─────────────────────────────────────────
  // BACKGROUND
  // ─────────────────────────────────────────
  _drawBackground(cx, cy) {
    const { ctx, opts, palette, t } = this;
    const W = opts.width, H = opts.height;
    const bg = palette.bg;

    const grad = ctx.createRadialGradient(cx, cy, 20, cx, H * 0.5, H * 0.8);
    grad.addColorStop(0, `rgba(${Math.min(bg[0]+15,255)},${Math.min(bg[1]+10,255)},${Math.min(bg[2]+20,255)},1)`);
    grad.addColorStop(1, `rgba(${bg[0]},${bg[1]},${bg[2]},1)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Stage-specific bg effects
    if (this.stage === 1 || this.stage === 2) {
      // Underwater caustic shimmer
      for (let i = 0; i < 4; i++) {
        const rx = 10 + i * (W / 4);
        const alpha = 0.015 + Math.sin(t * 0.4 + i) * 0.008;
        ctx.beginPath();
        ctx.moveTo(rx, 0);
        ctx.lineTo(rx + 15, H); ctx.lineTo(rx + 28, H); ctx.lineTo(rx + 13, 0);
        ctx.fillStyle = `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${alpha})`;
        ctx.fill();
      }
    }

    if (this.stage === 4) {
      // Occasional lightning flash bg
      if (Math.sin(t * 7) > 0.94) {
        ctx.fillStyle = `rgba(${palette.body[0]},${palette.body[1]},${palette.body[2]},0.04)`;
        ctx.fillRect(0, 0, W, H);
      }
    }

    if (this.stage === 5) {
      // Star field
      for (const s of this._stars) {
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,240,200,${0.15 + Math.sin(t + s.phase) * 0.08})`;
        ctx.fill();
      }
    }
  }

  // ─────────────────────────────────────────
  // PARTICLES (ambient)
  // ─────────────────────────────────────────
  _drawParticles(cx, cy) {
    const { ctx, opts, palette, t, data } = this;
    const W = opts.width, H = opts.height;

    // LLM bubble colours by provider
    const providerColors = {
      deepseek:  [80, 160, 255],
      kimi:      [80, 220, 150],
      anthropic: [255, 160, 80],
      unknown:   palette.particle,
    };

    for (const p of this._particles) {
      p.x  += p.vx + Math.sin(t * 0.5 + p.phase) * 0.3;
      p.y  += p.vy;
      if (p.y < -10) { p.y = H + 10; p.x = Math.random() * W; }
      if (p.x < 0 || p.x > W) p.x = Math.random() * W;

      // If recent LLM call, tint some particles
      const provs = Object.keys(data.llmCosts.byProvider);
      const tintColor = this._hasEffect('llm_bubble') && provs.length
        ? (providerColors[provs[provs.length - 1]] || palette.particle)
        : palette.particle;

      const a = (0.25 + Math.sin(t + p.phase) * 0.12) * p.alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${tintColor[0]},${tintColor[1]},${tintColor[2]},${a})`;
      ctx.fill();
    }
  }

  // ─────────────────────────────────────────
  // ORBIT PARTICLES (Storm stage)
  // ─────────────────────────────────────────
  _drawOrbitParticles(cx, cy) {
    const { ctx, t, palette, data } = this;
    const speedMult = 1 + (data.winStreak * 0.1);
    const countMult = Math.min(1, data.winStreak / 5);

    for (let i = 0; i < this._orbitPts.length; i++) {
      const p = this._orbitPts[i];
      if (i / this._orbitPts.length > 0.4 + countMult * 0.6) continue;
      p.angle += p.speed * speedMult;
      const px = cx + Math.cos(p.angle) * p.r;
      const py = cy + Math.sin(p.angle) * p.r * 0.65;
      // Trail
      for (let trail = 4; trail > 0; trail--) {
        const ta = p.angle - p.speed * trail * 2;
        const tx = cx + Math.cos(ta) * p.r;
        const ty = cy + Math.sin(ta) * p.r * 0.65;
        ctx.beginPath();
        ctx.arc(tx, ty, p.size * (trail / 4), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${palette.particle[0]},${palette.particle[1]},${palette.particle[2]},${0.12 * trail / 4})`;
        ctx.fill();
      }
      const a = 0.5 + Math.sin(t * 3 + p.phase) * 0.3;
      ctx.beginPath();
      ctx.arc(px, py, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${palette.accent[0]},${palette.accent[1]},${palette.accent[2]},${a})`;
      ctx.fill();
    }
  }

  // ─────────────────────────────────────────
  // NEBULA (Apex stage)
  // ─────────────────────────────────────────
  _drawNebula(cx, cy) {
    const { ctx, t, palette } = this;
    for (let layer = 0; layer < 3; layer++) {
      const r = 70 + layer * 28;
      const ng = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      ng.addColorStop(0,   `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${0.02 - layer * 0.005})`);
      ng.addColorStop(0.6, `rgba(255,100,30,0.008)`);
      ng.addColorStop(1,   `rgba(255,80,20,0)`);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = ng;
      ctx.fill();
    }
  }

  // ─────────────────────────────────────────
  // REGRESSION OVERLAY
  // ─────────────────────────────────────────
  _drawRegressionOverlay(cx, cy) {
    const { ctx, opts } = this;
    const progress = this._effectProgress('circuit_breaker');
    const alpha = 0.15 + Math.sin(this.t * 3) * 0.05;
    ctx.fillStyle = `rgba(180,170,190,${alpha})`;
    ctx.fillRect(0, 0, opts.width, opts.height);
  }

  // ─────────────────────────────────────────
  // EFFECT OVERLAYS
  // ─────────────────────────────────────────
  _drawEffectOverlays(cx, cy) {
    const { ctx, opts, palette } = this;

    // Trade executed flash
    if (this._hasEffect('trade_executed')) {
      const p = this._effectProgress('trade_executed');
      const alpha = (1 - p) * 0.25;
      ctx.fillStyle = `rgba(0,255,150,${alpha})`;
      ctx.fillRect(0, 0, opts.width, opts.height);
    }

    // Devil killed flash
    if (this._hasEffect('devil_killed')) {
      const p = this._effectProgress('devil_killed');
      ctx.fillStyle = `rgba(255,50,80,${(1-p)*0.2})`;
      ctx.fillRect(0, 0, opts.width, opts.height);
    }

    // Circuit breaker
    if (this._hasEffect('circuit_breaker')) {
      const p = this._effectProgress('circuit_breaker');
      ctx.fillStyle = `rgba(255,30,60,${(1-p)*0.35})`;
      ctx.fillRect(0, 0, opts.width, opts.height);
    }

    // Level up shimmer
    if (this._hasEffect('level_up')) {
      const p = this._effectProgress('level_up');
      const alpha = Math.sin(p * Math.PI) * 0.3;
      ctx.fillStyle = `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${alpha})`;
      ctx.fillRect(0, 0, opts.width, opts.height);
    }

    // Evolve burst
    if (this._hasEffect('evolve')) {
      const p = this._effectProgress('evolve');
      const r = p * opts.width;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0,   `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},0)`);
      grad.addColorStop(0.4, `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${0.4 * (1-p)})`);
      grad.addColorStop(1,   `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},0)`);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  // ═════════════════════════════════════════
  // STAGE RENDERERS
  // ═════════════════════════════════════════

  // ─────────────────────────────────────────
  // STAGE 0: DORMANT EGG (pre-hatching)
  // ─────────────────────────────────────────
  _drawEgg(cx, cy) {
    const { ctx, t, palette, cracks, eggGlow } = this;
    const PI2 = Math.PI * 2;

    const crackCount = cracks.length;
    // Wobble increases with each crack
    const wobble = Math.sin(t * 2) * (0.02 + crackCount * 0.005);
    const pulse  = 1 + Math.sin(t * 1.8) * (0.03 + eggGlow * 0.02);
    // Knock effect from inside
    const knock  = this._hasEffect('egg_pulse') ? Math.sin(this._effectProgress('egg_pulse') * Math.PI) * 0.04 : 0;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(wobble + knock);

    const eW = 52 * pulse, eH = 64 * pulse;

    // ── Outer glow (intensity = eggGlow) ──
    for (let r = 90; r > 0; r -= 12) {
      const a = (r / 90) * eggGlow * 0.06;
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      g.addColorStop(0, `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${a * 2})`);
      g.addColorStop(1, `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},0)`);
      ctx.beginPath(); ctx.ellipse(0, 0, r * 0.85, r, 0, 0, PI2);
      ctx.fillStyle = g; ctx.fill();
    }

    // ── Inner warm glow through cracks ──
    const glowA = eggGlow * 0.4 + Math.sin(t * 2) * eggGlow * 0.1;
    const innerG = ctx.createRadialGradient(0, -8, 4, 0, 0, eW * 0.85);
    innerG.addColorStop(0, `rgba(255,200,240,${glowA})`);
    innerG.addColorStop(0.6, `rgba(${palette.body[0]},${palette.body[1]},${palette.body[2]},${glowA * 0.4})`);
    innerG.addColorStop(1, `rgba(${palette.body[0]},${palette.body[1]},${palette.body[2]},0)`);
    ctx.beginPath(); ctx.ellipse(0, 0, eW, eH, 0, 0, PI2);
    ctx.fillStyle = innerG; ctx.fill();

    // ── Larva shadow visible inside ──
    const shadowAlpha = 0.08 + eggGlow * 0.14 + Math.sin(t * 1.3) * 0.04;
    const shadowWiggle = Math.sin(t * 2.5) * 4; // larva moving inside

    ctx.globalAlpha = shadowAlpha;
    // Body blob
    ctx.beginPath();
    ctx.ellipse(shadowWiggle * 0.3, 10, 18, 22, 0, 0, PI2);
    ctx.fillStyle = `rgba(180,140,220,0.7)`; ctx.fill();
    // Head blob
    ctx.beginPath();
    ctx.arc(shadowWiggle * 0.5, -14, 15, 0, PI2);
    ctx.fillStyle = `rgba(200,160,240,0.7)`; ctx.fill();
    // Tiny gill nubs
    for (let side of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(side * 14 + shadowWiggle * 0.3, -20, 4, 9, side * 0.3, 0, PI2);
      ctx.fillStyle = `rgba(220,160,200,0.5)`; ctx.fill();
    }

    // When enough cracks, larva EYE peeks through largest crack
    if (crackCount >= 3 || eggGlow > 0.6) {
      ctx.globalAlpha = Math.min(1, (eggGlow - 0.4) * 2);
      // One glowing eye visible
      const eyeX = shadowWiggle * 0.4 - 6;
      const eyeY = -14;
      const eyeG = ctx.createRadialGradient(eyeX, eyeY, 0, eyeX, eyeY, 7);
      eyeG.addColorStop(0, `rgba(220,180,255,0.9)`);
      eyeG.addColorStop(0.4, `rgba(${palette.eye[0]},${palette.eye[1]},${palette.eye[2]},0.7)`);
      eyeG.addColorStop(1, `rgba(${palette.eye[0]},${palette.eye[1]},${palette.eye[2]},0)`);
      ctx.beginPath(); ctx.arc(eyeX, eyeY, 7, 0, PI2);
      ctx.fillStyle = eyeG; ctx.fill();
      // Pupil
      ctx.globalAlpha = Math.min(0.8, (eggGlow - 0.4) * 2);
      ctx.beginPath(); ctx.arc(eyeX, eyeY, 3, 0, PI2);
      ctx.fillStyle = `rgba(5,3,15,0.9)`; ctx.fill();
    }

    // Tiny paw print pressed against shell (when glow > 0.5)
    if (eggGlow > 0.5) {
      ctx.globalAlpha = (eggGlow - 0.5) * 0.8;
      const pawX = shadowWiggle * 0.2 + 10, pawY = 8;
      ctx.beginPath(); ctx.arc(pawX, pawY, 5, 0, PI2);
      ctx.fillStyle = `rgba(200,160,230,0.5)`; ctx.fill();
      for (let toe = 0; toe < 3; toe++) {
        const ta = ((toe / 3) * Math.PI) - Math.PI / 2 - 0.2;
        ctx.beginPath(); ctx.arc(pawX + Math.cos(ta) * 7, pawY + Math.sin(ta) * 6, 2.5, 0, PI2);
        ctx.fillStyle = `rgba(200,160,230,0.4)`; ctx.fill();
      }
    }

    ctx.globalAlpha = 1;

    // ── Egg shell ──
    const shellG = ctx.createRadialGradient(-eW * 0.25, -eH * 0.3, 4, 0, 0, eW);
    shellG.addColorStop(0, `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},0.2)`);
    shellG.addColorStop(0.5, `rgba(${palette.body[0]},${palette.body[1]},${palette.body[2]},0.12)`);
    shellG.addColorStop(1, `rgba(${palette.body[0]},${palette.body[1]},${palette.body[2]},0.04)`);
    ctx.beginPath(); ctx.ellipse(0, 0, eW, eH, 0, 0, PI2);
    ctx.fillStyle = shellG; ctx.fill();

    // Shell rim
    ctx.beginPath(); ctx.ellipse(0, 0, eW, eH, 0, 0, PI2);
    ctx.strokeStyle = `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},${0.3 + eggGlow * 0.3})`;
    ctx.lineWidth = 1.5; ctx.stroke();

    // Highlight
    const hiG = ctx.createRadialGradient(-eW * 0.3, -eH * 0.35, 0, -eW * 0.3, -eH * 0.35, eW * 0.5);
    hiG.addColorStop(0, `rgba(255,255,255,0.12)`);
    hiG.addColorStop(1, `rgba(255,255,255,0)`);
    ctx.beginPath(); ctx.ellipse(0, 0, eW, eH, 0, 0, PI2);
    ctx.fillStyle = hiG; ctx.fill();

    // ── Cracks ──
    for (let i = 0; i < cracks.length; i++) {
      const crack = cracks[i];
      const glowLeak = ctx.createRadialGradient(crack.ox, crack.oy, 0, crack.ox, crack.oy, 12);
      glowLeak.addColorStop(0, `rgba(255,200,240,${0.5 * eggGlow})`);
      glowLeak.addColorStop(1, `rgba(255,200,240,0)`);
      ctx.beginPath(); ctx.arc(crack.ox, crack.oy, 12, 0, PI2);
      ctx.fillStyle = glowLeak; ctx.fill();

      // Main crack line
      ctx.beginPath();
      ctx.moveTo(crack.ox, crack.oy);
      const segments = 6;
      let cx2 = crack.ox, cy2 = crack.oy;
      for (let s = 0; s < segments; s++) {
        const f = s / segments;
        const jitter = (Math.random() - 0.5) * 4;
        cx2 += Math.cos(crack.startAngle + jitter * 0.1) * (crack.len / segments);
        cy2 += Math.sin(crack.startAngle) * (crack.len / segments) + jitter;
        ctx.lineTo(cx2, cy2);
      }
      ctx.strokeStyle = `rgba(255,220,250,${0.7 + eggGlow * 0.3})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    // ── Hatch sequence ──
    if (this._hasEffect('hatch')) {
      const p = this._effectProgress('hatch');
      // Shell pieces fly apart
      for (let s = 0; s < 6; s++) {
        const sa = (s / 6) * PI2 + p * 2;
        const sd = p * 80;
        const sx = Math.cos(sa) * sd, sy = Math.sin(sa) * sd;
        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(sa + p * 3);
        ctx.globalAlpha = 1 - p;
        ctx.beginPath();
        ctx.arc(0, 0, 10 + s * 3, 0, PI2 * 0.4);
        ctx.strokeStyle = `rgba(${palette.glow[0]},${palette.glow[1]},${palette.glow[2]},0.8)`;
        ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
      }
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  }

  // ─────────────────────────────────────────
  // STAGE 1: LARVA
  // ─────────────────────────────────────────
  _drawLarva(cx, cy) {
    const { ctx, t, palette, mood } = this;
    const PI2 = Math.PI * 2;

    const isExcited = this._hasEffect('trade_executed') || this._hasEffect('devil_approved');
    const isSad     = mood === 'sad' || this._hasEffect('devil_killed');

    const bob     = Math.sin(t * 2) * (isExcited ? 8 : 4);
    const wiggle  = Math.sin(t * (isExcited ? 5 : 3)) * (isExcited ? 0.15 : 0.07);
    const spinPct = this._hasEffect('trade_executed') ? this._effectProgress('trade_executed') : 0;

    const B = palette.body, G = palette.glow, E = palette.eye;

    ctx.save();
    ctx.translate(cx, cy + bob);
    if (spinPct > 0) ctx.rotate(spinPct * Math.PI * 2);

    // Tail
    ctx.beginPath(); ctx.moveTo(0, 20);
    for (let i = 1; i <= 10; i++) {
      const f = i / 10;
      ctx.lineTo(Math.sin(t * 2 + f * 3) * 12 * f, 20 + f * 32);
    }
    const tg = ctx.createLinearGradient(0, 20, 0, 52);
    tg.addColorStop(0, `rgba(${B[0]},${B[1]},${B[2]},0.8)`);
    tg.addColorStop(1, `rgba(${B[0]},${B[1]},${B[2]},0.05)`);
    ctx.strokeStyle = tg; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.stroke();

    // Body
    const bG = ctx.createRadialGradient(-2, 0, 3, 0, 4, 22);
    bG.addColorStop(0, `rgba(${G[0]},${G[1]},${G[2]},0.95)`);
    bG.addColorStop(0.5, `rgba(${B[0]},${B[1]},${B[2]},0.85)`);
    bG.addColorStop(1, `rgba(${Math.max(0,B[0]-40)},${Math.max(0,B[1]-40)},${Math.max(0,B[2]-40)},0.7)`);
    ctx.beginPath(); ctx.ellipse(0, 6, 18, 22, 0, 0, PI2);
    ctx.fillStyle = bG; ctx.fill();

    // Stubby legs
    for (let side of [-1, 1]) {
      ctx.beginPath(); ctx.ellipse(side * 16, 14, 7, 4, side * 0.4, 0, PI2);
      ctx.fillStyle = `rgba(${B[0]},${B[1]},${B[2]},0.8)`; ctx.fill();
    }
    ctx.restore();

    // Head
    ctx.save();
    ctx.translate(cx, cy - 18 + bob);
    ctx.rotate(wiggle);

    const hG = ctx.createRadialGradient(0, -2, 2, 0, 2, 22);
    hG.addColorStop(0, `rgba(${G[0]},${G[1]},${G[2]},1)`);
    hG.addColorStop(0.4, `rgba(${B[0]},${B[1]},${B[2]},0.9)`);
    hG.addColorStop(1, `rgba(${Math.max(0,B[0]-40)},${Math.max(0,B[1]-40)},${Math.max(0,B[2]-40)},0.7)`);
    ctx.beginPath(); ctx.ellipse(0, 0, 22, 20, 0, 0, PI2);
    ctx.fillStyle = hG; ctx.fill();

    // Stubby gills — perk on scheduler task
    const gillPerk = this._hasEffect('gill_perk') ? 1.3 : 1;
    for (let side of [-1, 1]) {
      for (let g = 0; g < 2; g++) {
        const gx = side * 20, gy = -8 - g * 8;
        const len = (12 - g * 3) * gillPerk;
        const wave = Math.sin(t * 1.5 + g) * 4;
        ctx.beginPath(); ctx.moveTo(gx, gy);
        ctx.quadraticCurveTo(gx + side * wave, gy - len * 0.5, gx + side * 2, gy - len);
        ctx.strokeStyle = `rgba(${B[0]},${B[1]},${B[2]},0.7)`;
        ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.stroke();
        const tG = ctx.createRadialGradient(gx + side * 2, gy - len, 0, gx + side * 2, gy - len, 4);
        tG.addColorStop(0, `rgba(${G[0]},${G[1]},${G[2]},0.8)`);
        tG.addColorStop(1, `rgba(${G[0]},${G[1]},${G[2]},0)`);
        ctx.beginPath(); ctx.arc(gx + side * 2, gy - len, 4, 0, PI2);
        ctx.fillStyle = tG; ctx.fill();
      }
    }

    // Snout
    ctx.beginPath(); ctx.ellipse(0, 10, 8, 7, 0, 0, PI2);
    ctx.fillStyle = `rgba(${G[0]},${G[1]},${G[2]},0.7)`; ctx.fill();

    // Nostrils
    for (let s of [-1, 1]) {
      ctx.beginPath(); ctx.arc(s * 3, 13, 1.5, 0, PI2);
      ctx.fillStyle = `rgba(${Math.max(0,B[0]-80)},${Math.max(0,B[1]-80)},${Math.max(0,B[2]-60)},0.7)`;
      ctx.fill();
    }

    // Mouth
    if (isSad) {
      ctx.beginPath(); ctx.moveTo(-5, 18); ctx.quadraticCurveTo(0, 14, 5, 18);
    } else {
      ctx.beginPath(); ctx.moveTo(-5, 16); ctx.quadraticCurveTo(0, 20, 5, 16);
    }
    ctx.strokeStyle = `rgba(${Math.max(0,B[0]-80)},${Math.max(0,B[1]-80)},${Math.max(0,B[2]-60)},0.6)`;
    ctx.lineWidth = 1.5; ctx.lineCap = 'round'; ctx.stroke();

    // Look-around eye movement
    const lookOffset = this._hasEffect('look_around')
      ? Math.sin(this._effectProgress('look_around') * Math.PI * 2) * 3 : 0;

    // Eyes
    for (let side of [-1, 1]) {
      const ex = side * 9 + lookOffset, ey = -4;
      ctx.beginPath(); ctx.arc(ex, ey, 8, 0, PI2);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fill();

      const iG = ctx.createRadialGradient(ex, ey, 0, ex, ey, 5);
      iG.addColorStop(0, `rgba(${E[0]},${E[1]},${E[2]},1)`);
      iG.addColorStop(1, `rgba(${Math.max(0,E[0]-60)},${Math.max(0,E[1]-60)},${Math.max(0,E[2]-40)},1)`);
      ctx.beginPath(); ctx.arc(ex, ey, 5, 0, PI2); ctx.fillStyle = iG; ctx.fill();

      // Star pupils when excited
      if (isExcited) {
        ctx.fillStyle = 'rgba(5,5,15,0.9)';
        for (let si = 0; si < 4; si++) {
          const sa = (si / 4) * PI2;
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex + Math.cos(sa) * 4, ey + Math.sin(sa) * 4);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = 'rgba(5,5,15,0.9)';
          ctx.stroke();
        }
      } else {
        ctx.beginPath(); ctx.arc(ex, ey, 2.5, 0, PI2);
        ctx.fillStyle = 'rgba(5,5,15,0.95)'; ctx.fill();
      }
      ctx.beginPath(); ctx.arc(ex - 2, ey - 2, 2, 0, PI2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
    }

    // Cheeks
    for (let side of [-1, 1]) {
      const cg = ctx.createRadialGradient(side * 17, 5, 0, side * 17, 5, 8);
      cg.addColorStop(0, `rgba(${B[0]},${Math.min(255,B[1]+30)},${B[2]},0.35)`);
      cg.addColorStop(1, `rgba(${B[0]},${B[1]},${B[2]},0)`);
      ctx.beginPath(); ctx.arc(side * 17, 5, 8, 0, PI2);
      ctx.fillStyle = cg; ctx.fill();
    }

    ctx.restore();
  }

  // ─────────────────────────────────────────
  // STAGE 2: LOTUS (full pink form)
  // ─────────────────────────────────────────
  _drawLotus(cx, cy) {
    const { ctx, t, palette, mood, data } = this;
    const PI2 = Math.PI * 2;

    const B = palette.body, G = palette.glow, E = palette.eye, A = palette.accent;

    // Mood shifts body colour
    let bodyTint = [...B];
    if (mood === 'happy' || data.dailyPnl > 1) {
      // Warmer — shift toward coral
      bodyTint = [Math.min(255, B[0] + 20), Math.max(0, B[1] - 20), Math.max(0, B[2] - 30)];
    } else if (mood === 'sad' || data.dailyPnl < -0.5) {
      // Cooler — shift toward lavender
      bodyTint = [Math.max(0, B[0] - 30), Math.max(0, B[1] - 20), Math.min(255, B[2] + 40)];
    }

    const isWilting  = data.drawdown > 5 || mood === 'sad';
    const isGlowing  = this._hasEffect('thesis') && (this.effects.find(e => e.type === 'thesis')?.data.confidence || 0) > 0.8;
    const isStartled = this._hasEffect('devil_killed');
    const hasSparks  = data.winStreak >= 3;

    const bob     = Math.sin(t * (isWilting ? 0.8 : 1.4)) * (isWilting ? 2 : 4);
    const squish  = 1 + Math.sin(t * 1.4) * (isWilting ? 0.01 : 0.025);
    const startleOffset = isStartled ? Math.sin(this._effectProgress('devil_killed') * Math.PI) * -15 : 0;

    // Win streak sparkles orbiting body
    if (hasSparks) {
      const sparkCount = Math.min(8, data.winStreak);
      for (let i = 0; i < sparkCount; i++) {
        const sa = (i / sparkCount) * PI2 + t * 1.5;
        const sr = 55 + Math.sin(t * 2 + i) * 8;
        const sx = cx + Math.cos(sa) * sr;
        const sy = cy + bob + Math.sin(sa) * sr * 0.5;
        const sparklePulse = 0.4 + Math.sin(t * 3 + i * 0.8) * 0.3;
        const spG = ctx.createRadialGradient(sx, sy, 0, sx, sy, 5);
        spG.addColorStop(0, `rgba(${A[0]},${A[1]},${A[2]},${sparklePulse})`);
        spG.addColorStop(1, `rgba(${A[0]},${A[1]},${A[2]},0)`);
        ctx.beginPath(); ctx.arc(sx, sy, 5, 0, PI2);
        ctx.fillStyle = spG; ctx.fill();
      }
    }

    ctx.save();
    ctx.translate(cx, cy + bob + startleOffset);

    // Tail
    ctx.beginPath(); ctx.moveTo(0, 20);
    for (let i = 1; i <= 12; i++) {
      const f = i / 12;
      ctx.lineTo(Math.sin(t * 1.5 + f * 4) * 14 * f, 20 + f * 38);
    }
    const tg = ctx.createLinearGradient(0, 20, 0, 58);
    tg.addColorStop(0, `rgba(${bodyTint[0]},${bodyTint[1]},${bodyTint[2]},0.8)`);
    tg.addColorStop(1, `rgba(${bodyTint[0]},${bodyTint[1]},${bodyTint[2]},0.05)`);
    ctx.strokeStyle = tg; ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.stroke();

    // Body
    const bG = ctx.createRadialGradient(0, 0, 4, 0, 5, 32);
    bG.addColorStop(0, `rgba(${A[0]},${A[1]},${A[2]},0.95)`);
    bG.addColorStop(0.4, `rgba(${bodyTint[0] + 30 > 255 ? 255 : bodyTint[0] + 30},${bodyTint[1] + 20 > 255 ? 255 : bodyTint[1] + 20},${bodyTint[2] + 20 > 255 ? 255 : bodyTint[2] + 20},0.9)`);
    bG.addColorStop(1, `rgba(${bodyTint[0]},${bodyTint[1]},${bodyTint[2]},0.7)`);
    ctx.beginPath(); ctx.ellipse(0, 8, 26 * squish, 30, 0, 0, PI2);
    ctx.fillStyle = bG; ctx.fill();

    // Spots
    for (let i = 0; i < 6; i++) {
      const sx = (i % 3 - 1) * 14 + Math.sin(i) * 4;
      const sy = Math.floor(i / 3) * 16 - 5;
      ctx.beginPath(); ctx.arc(sx, sy, 4 + Math.sin(i) * 1, 0, PI2);
      ctx.fillStyle = `rgba(${E[0]},${E[1]},${E[2]},0.12)`; ctx.fill();
    }

    // Legs
    for (let side of [-1, 1]) {
      ctx.beginPath(); ctx.ellipse(side * 22, 18, 9, 6, side * 0.5, 0, PI2);
      ctx.fillStyle = `rgba(${A[0]},${A[1]},${A[2]},0.8)`; ctx.fill();
    }

    ctx.restore();

    // Head
    ctx.save();
    ctx.translate(cx, cy - 24 + bob + startleOffset);

    const hG = ctx.createRadialGradient(0, 0, 3, 0, 3, 24);
    hG.addColorStop(0, `rgba(${A[0]},${A[1]},${A[2]},1)`);
    hG.addColorStop(0.4, `rgba(${bodyTint[0]+20 > 255 ? 255 : bodyTint[0]+20},${bodyTint[1]+15 > 255 ? 255 : bodyTint[1]+15},${bodyTint[2]+15 > 255 ? 255 : bodyTint[2]+15},0.95)`);
    hG.addColorStop(1, `rgba(${bodyTint[0]},${bodyTint[1]},${bodyTint[2]},0.75)`);
    ctx.beginPath(); ctx.ellipse(0, 2, 22, 20, 0, 0, PI2);
    ctx.fillStyle = hG; ctx.fill();

    // Gills — fan wider when happy/high-conf, droop when sad
    const gillScale = isWilting ? 0.7 : (isGlowing ? 1.3 : 1.0);
    const gillDroop = isWilting ? 0.4 : 0;
    this._drawLotusGills(ctx, t, B, G, A, gillScale, gillDroop, isGlowing);

    // Thesis head-tilt
    if (this._hasEffect('thesis')) {
      const conf = this.effects.find(e => e.type === 'thesis')?.data.confidence || 0;
      if (conf < 0.5) {
        ctx.rotate(0.15 * Math.sin(this._effectProgress('thesis') * Math.PI));
      }
    }

    // Mouth
    if (mood === 'sad') {
      ctx.beginPath(); ctx.moveTo(-6, 12); ctx.quadraticCurveTo(0, 8, 6, 12);
    } else if (mood === 'happy' || mood === 'excited') {
      ctx.beginPath(); ctx.moveTo(-8, 10); ctx.quadraticCurveTo(0, 18, 8, 10);
    } else {
      ctx.beginPath(); ctx.moveTo(-6, 12); ctx.quadraticCurveTo(0, 17, 6, 12);
    }
    ctx.strokeStyle = `rgba(${Math.max(0,B[0]-80)},${Math.max(0,B[1]-80)},${Math.max(0,B[2]-60)},0.5)`;
    ctx.lineWidth = 1.5; ctx.lineCap = 'round'; ctx.stroke();

    // Sleeping Zzz
    if (this.sleeping) {
      for (let z = 0; z < 3; z++) {
        const zy = -30 - z * 14 + Math.sin(t * 0.5 + z) * 3;
        const za = 0.5 - z * 0.12;
        ctx.font = `bold ${10 + z * 3}px sans-serif`;
        ctx.fillStyle = `rgba(${B[0]},${B[1]},${B[2]},${za})`;
        ctx.fillText('z', 22 + z * 5, zy);
      }
    }

    // Nostrils
    for (let s of [-1, 1]) {
      ctx.beginPath(); ctx.arc(s * 4, 7, 1.8, 0, PI2);
      ctx.fillStyle = `rgba(${Math.max(0,B[0]-80)},${Math.max(0,B[1]-80)},${Math.max(0,B[2]-60)},0.4)`;
      ctx.fill();
    }

    // Eyes
    const lookOffset = this._hasEffect('look_around') || this._hasEffect('news_scan')
      ? Math.sin(this._effectProgress('look_around') || this._effectProgress('news_scan') * Math.PI * 2) * 3 : 0;

    for (let side of [-1, 1]) {
      const ex = side * 10 + lookOffset, ey = -2;
      ctx.beginPath(); ctx.arc(ex, ey, 8, 0, PI2);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fill();

      const iG = ctx.createRadialGradient(ex, ey, 0, ex, ey, 5.5);
      iG.addColorStop(0, `rgba(${E[0]},${E[1]},${E[2]},1)`);
      iG.addColorStop(1, `rgba(${Math.max(0,E[0]-60)},${Math.max(0,E[1]-60)},${Math.max(0,E[2]-40)},1)`);
      ctx.beginPath(); ctx.arc(ex, ey, 5.5, 0, PI2); ctx.fillStyle = iG; ctx.fill();

      // Sleepy half-lids
      if (this.sleeping) {
        ctx.beginPath(); ctx.rect(ex - 9, ey - 9, 18, 8);
        ctx.fillStyle = `rgba(${bodyTint[0]},${bodyTint[1]},${bodyTint[2]},0.85)`; ctx.fill();
      }

      ctx.beginPath(); ctx.arc(ex, ey, 2.8, 0, PI2);
      ctx.fillStyle = 'rgba(5,3,8,0.95)'; ctx.fill();
      ctx.beginPath(); ctx.arc(ex - 2, ey - 2, 2, 0, PI2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fill();
      ctx.beginPath(); ctx.arc(ex + 2, ey + 1.5, 1.2, 0, PI2);
      ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.fill();

      // Widen eyes on high-conf thesis
      if (isGlowing) {
        const eyeGlow = ctx.createRadialGradient(ex, ey, 0, ex, ey, 10);
        eyeGlow.addColorStop(0, `rgba(${G[0]},${G[1]},${G[2]},0.3)`);
        eyeGlow.addColorStop(1, `rgba(${G[0]},${G[1]},${G[2]},0)`);
        ctx.beginPath(); ctx.arc(ex, ey, 10, 0, PI2);
        ctx.fillStyle = eyeGlow; ctx.fill();
      }
    }

    // Cheeks
    for (let side of [-1, 1]) {
      const cg = ctx.createRadialGradient(side * 16, 4, 0, side * 16, 4, 9);
      cg.addColorStop(0, `rgba(${B[0]},${Math.min(255,B[1]+40)},${B[2]},${mood === 'happy' ? 0.5 : 0.35})`);
      cg.addColorStop(1, `rgba(${B[0]},${B[1]},${B[2]},0)`);
      ctx.beginPath(); ctx.arc(side * 16, 4, 9, 0, PI2);
      ctx.fillStyle = cg; ctx.fill();
    }

    ctx.restore();
  }

  _drawLotusGills(ctx, t, B, G, A, scale, droop, isGlowing) {
    const fronds = 4;
    const gillDefs = [
      { x: -20, y: -6, angle: -2.4, phase: 0 },
      { x:  20, y: -6, angle: -0.7, phase: 1.5 },
      { x:   0, y: -18, angle: -1.57, phase: 0.8 },
    ];

    for (const g of gillDefs) {
      for (let i = 0; i < fronds; i++) {
        const fa = g.angle + (i - (fronds - 1) / 2) * 0.25;
        const len = (30 - Math.abs(i - (fronds - 1) / 2) * 4) * scale;
        const wave = Math.sin(t * 1.3 + g.phase + i * 0.5) * 7;
        const droopOffset = droop * len * 0.3;
        const ex = g.x + Math.cos(fa) * len + wave;
        const ey = g.y + Math.sin(fa) * len - 6 + droopOffset;

        ctx.beginPath(); ctx.moveTo(g.x, g.y);
        ctx.quadraticCurveTo(
          g.x + Math.cos(fa) * len * 0.5 + wave * 0.5,
          g.y + Math.sin(fa) * len * 0.5 - 3 + droopOffset * 0.5,
          ex, ey
        );
        const grad = ctx.createLinearGradient(g.x, g.y, ex, ey);
        grad.addColorStop(0, `rgba(${B[0]},${B[1]},${B[2]},0.85)`);
        grad.addColorStop(1, `rgba(${B[0]},${B[1]},${B[2]},0.1)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = (4 - Math.abs(i - (fronds - 1) / 2) * 0.5) * scale;
        ctx.lineCap = 'round'; ctx.stroke();

        // Glow tip
        const tintColor = isGlowing ? G : B;
        const tG = ctx.createRadialGradient(ex, ey, 0, ex, ey, 5 * scale);
        tG.addColorStop(0, `rgba(${tintColor[0]},${tintColor[1]},${tintColor[2]},${isGlowing ? 0.9 : 0.5})`);
        tG.addColorStop(1, `rgba(${tintColor[0]},${tintColor[1]},${tintColor[2]},0)`);
        ctx.beginPath(); ctx.arc(ex, ey, 5 * scale, 0, Math.PI * 2);
        ctx.fillStyle = tG; ctx.fill();
      }
    }
  }

  // ─────────────────────────────────────────
  // STAGE 3: ELDER
  // ─────────────────────────────────────────
  _drawElder(cx, cy) {
    const { ctx, t, palette } = this;
    // Elder shares Lotus base, with violet palette + extra features
    // Rune flash on trade
    const runeAlpha = this._hasEffect('trade_executed') ? 0.5 + Math.sin(t * 20) * 0.3 : 0.2;
    // Re-use lotus draw with elder palette + add third eye + runes
    this._drawLotus(cx, cy);
    this._drawElderExtras(cx, cy, runeAlpha);
  }

  _drawElderExtras(cx, cy, runeAlpha) {
    const { ctx, t, palette } = this;
    const PI2 = Math.PI * 2;
    const E = palette.eye, G = palette.glow;
    const bob = Math.sin(t * 1.1) * 3;

    ctx.save();
    ctx.translate(cx, cy - 24 + bob);

    // Rune markings on face
    ctx.strokeStyle = `rgba(${G[0]},${G[1]},${G[2]},${runeAlpha})`;
    ctx.lineWidth = 0.8;
    for (let side of [-1, 1]) {
      ctx.beginPath(); ctx.moveTo(side * 5, 14); ctx.lineTo(side * 18, 16); ctx.stroke();
    }

    // Crystal third eye
    const tp = 1 + Math.sin(t * 3) * 0.15;
    const teG = ctx.createRadialGradient(0, -14, 0, 0, -14, 8 * tp);
    teG.addColorStop(0, `rgba(${E[0]},${E[1]},${E[2]},0.95)`);
    teG.addColorStop(0.4, `rgba(${G[0]},${G[1]},${G[2]},0.6)`);
    teG.addColorStop(1, `rgba(${G[0]},${G[1]},${G[2]},0)`);
    ctx.beginPath(); ctx.ellipse(0, -14, 8 * tp, 6 * tp, 0, 0, PI2);
    ctx.fillStyle = teG; ctx.fill();

    ctx.shadowColor = `rgba(${E[0]},${E[1]},${E[2]},0.8)`;
    ctx.shadowBlur = 10 + Math.sin(t * 3) * 5;
    ctx.beginPath(); ctx.arc(0, -14, 2.5, 0, PI2);
    ctx.fillStyle = `rgba(${E[0]},${E[1]},${E[2]},1)`; ctx.fill();
    ctx.shadowBlur = 0;

    ctx.restore();
  }

  // ─────────────────────────────────────────
  // STAGE 4: STORM
  // ─────────────────────────────────────────
  _drawStorm(cx, cy) {
    // Storm uses elder as base, adds electric effects
    this._drawElder(cx, cy);
    this._drawStormExtras(cx, cy);
  }

  _drawStormExtras(cx, cy) {
    const { ctx, t, palette, data } = this;
    const PI2 = Math.PI * 2;
    const B = palette.body, A = palette.accent;
    const flicker = 0.85 + Math.sin(t * 18) * 0.15;
    const bob = Math.sin(t * 2.5) * 5;

    ctx.save();
    ctx.translate(cx, cy - 24 + bob);

    // Electric aura
    ctx.shadowColor = `rgba(${B[0]},${B[1]},${B[2]},0.6)`;
    ctx.shadowBlur = 18 * flicker;
    ctx.beginPath(); ctx.ellipse(0, 2, 24, 21, 0, 0, PI2);
    ctx.strokeStyle = `rgba(${B[0]},${B[1]},${B[2]},${0.2 * flicker})`;
    ctx.lineWidth = 2; ctx.stroke();
    ctx.shadowBlur = 0;

    // Lightning arcs between gill tips on high-conf thesis
    if (this._hasEffect('thesis')) {
      const conf = this.effects.find(e => e.type === 'thesis')?.data.confidence || 0;
      if (conf > 0.7) {
        ctx.beginPath();
        ctx.moveTo(-24, -20);
        for (let s = 0; s < 8; s++) {
          ctx.lineTo(-24 + s * 6 + Math.sin(t * 15 + s) * 5, -20 - Math.abs(Math.sin(t * 10 + s)) * 20);
        }
        ctx.lineTo(24, -20);
        ctx.strokeStyle = `rgba(${A[0]},${A[1]},${A[2]},0.6)`;
        ctx.lineWidth = 1; ctx.stroke();
      }
    }

    ctx.restore();
  }

  // ─────────────────────────────────────────
  // STAGE 5: APEX
  // ─────────────────────────────────────────
  _drawApex(cx, cy) {
    this._drawElder(cx, cy);
    this._drawApexExtras(cx, cy);
  }

  _drawApexExtras(cx, cy) {
    const { ctx, t, palette, data } = this;
    const PI2 = Math.PI * 2;
    const G = palette.glow, A = palette.accent;
    const bob = Math.sin(t * 1.3) * 3;
    const isATH = this._hasEffect('level_up');

    ctx.save();
    ctx.translate(cx, cy - 24 + bob);

    // Crown of flame-gills
    for (let i = 0; i < 5; i++) {
      const ca = (i / 5) * PI2 + t * 0.2;
      const cr = 18, ch = 8 + Math.sin(t * 2 + i) * 3 + (isATH ? 5 : 0);
      ctx.beginPath();
      ctx.moveTo(Math.cos(ca) * cr, -22 + Math.sin(ca) * cr * 0.4);
      ctx.lineTo(Math.cos(ca) * cr * 0.7, -22 + Math.sin(ca) * cr * 0.4 - ch);
      ctx.lineTo(Math.cos(ca) * cr * 0.4, -22 + Math.sin(ca) * cr * 0.4);
      const cG = ctx.createLinearGradient(0, -22, 0, -22 - ch);
      cG.addColorStop(0, `rgba(${G[0]},${G[1]},${G[2]},0.9)`);
      cG.addColorStop(1, `rgba(255,120,30,0.5)`);
      ctx.fillStyle = cG; ctx.fill();
    }

    // Scale shimmer on win
    if (this._hasEffect('trade_executed')) {
      const p = this._effectProgress('trade_executed');
      ctx.strokeStyle = `rgba(${G[0]},${G[1]},${G[2]},${(1 - p) * 0.5})`;
      ctx.lineWidth = 0.8;
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * PI2 + p * PI2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(a) * 20, Math.sin(a) * 16);
        ctx.stroke();
      }
    }

    // Third eye (golden)
    const tp = 1 + Math.sin(t * 2.5) * 0.2;
    ctx.shadowColor = `rgba(${G[0]},${G[1]},${G[2]},0.9)`;
    ctx.shadowBlur = 16 * tp;
    const teG = ctx.createRadialGradient(0, -14, 0, 0, -14, 9 * tp);
    teG.addColorStop(0, `rgba(255,255,200,1)`);
    teG.addColorStop(0.4, `rgba(${G[0]},${G[1]},${G[2]},0.8)`);
    teG.addColorStop(1, `rgba(${G[0]},${G[1]},${G[2]},0)`);
    ctx.beginPath(); ctx.ellipse(0, -14, 9 * tp, 6 * tp, 0, 0, PI2);
    ctx.fillStyle = teG; ctx.fill();
    ctx.beginPath(); ctx.arc(0, -14, 3, 0, PI2);
    ctx.fillStyle = `rgba(255,240,180,1)`; ctx.fill();
    ctx.shadowBlur = 0;

    ctx.restore();
  }

  // ═════════════════════════════════════════
  // PUBLIC GETTERS (for external HUD rendering)
  // ═════════════════════════════════════════

  getState() {
    return {
      stage:       this.stage,
      stageName:   ['Dormant','Larva','Lotus','Elder','Storm','Apex'][this.stage],
      level:       this.level,
      xp:          this.xp,
      xpToNext:    this.xpToNext,
      mood:        this.mood,
      regression:  this.regression,
      sleeping:    this.sleeping,
      winStreak:   this.data.winStreak,
      hp:          Math.max(0, 100 - (this.data.drawdown * 5)),
      palette:     this.palette,
    };
  }
}

// Export for module use OR expose as global
if (typeof module !== 'undefined') module.exports = { LotusCreature };
else window.LotusCreature = LotusCreature;
