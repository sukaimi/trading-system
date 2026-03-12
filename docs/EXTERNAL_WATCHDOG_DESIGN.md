# External VPS Watchdog — Design Document

**Date**: 2026-03-12
**Status**: Planning
**Author**: Sukaimi (Code&Canvas)

## Problem

The trading system runs on a single VPS at `https://tradebot.codeandcraft.ai`. If the VPS goes down (crash, network outage, provider issue), the system cannot send Telegram alerts because the alerting code runs on the same VPS. The owner has no way to know the system is offline until they manually check.

An **external** monitor running on separate infrastructure is required.

## Recommendation

**Primary: UptimeRobot (free tier)** — zero cost, zero maintenance, 5-minute checks.
**Optional: n8n workflow** — for richer logic (consecutive failure gating, custom alert formatting, future expansion).

UptimeRobot is recommended as the primary solution because it solves the core problem with no cost and minimal setup. n8n is documented below for users who want more control or plan to build additional automation workflows.

---

## Option A: UptimeRobot (Recommended)

### Why UptimeRobot

| Feature | UptimeRobot Free |
|---------|-------------------|
| Cost | $0/month |
| Monitors | 50 (we need 1) |
| Check interval | 5 minutes |
| Alert channels | Email, Telegram (via webhook), SMS, Slack, etc. |
| Uptime pages | Free status page included |
| Setup time | ~5 minutes |
| Maintenance | None — fully managed SaaS |

### Setup Instructions

1. **Create account**: Go to https://uptimerobot.com and sign up (free).

2. **Add monitor**:
   - Monitor Type: **HTTP(s)**
   - Friendly Name: `Tradebot Quant`
   - URL: `https://tradebot.codeandcraft.ai/api/portfolio`
   - Monitoring Interval: **5 minutes**
   - Monitor Timeout: **30 seconds**

3. **Configure alerts**:

   **Option 1 — Email (simplest)**: UptimeRobot sends email on down/up by default. Done.

   **Option 2 — Telegram (recommended)**: Use UptimeRobot's webhook alert contact to call the Telegram Bot API directly.

   - Go to **My Settings** > **Alert Contacts** > **Add Alert Contact**
   - Type: **Webhook**
   - Friendly Name: `Telegram Alert`
   - URL to Notify:
     ```
     https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/sendMessage
     ```
   - POST value (JSON):
     ```json
     {
       "chat_id": "<TELEGRAM_CHAT_ID>",
       "text": "Tradebot Watchdog: *monitorFriendlyName* is *alertTypeFriendlyName*\nURL: *monitorURL*\nDuration: *alertDuration*",
       "parse_mode": "Markdown"
     }
     ```
   - Enable **Send as JSON (application/json)**
   - Replace `<TELEGRAM_BOT_TOKEN>` and `<TELEGRAM_CHAT_ID>` with actual values from the VPS `.env` file.

4. **Attach alert to monitor**: Edit the Tradebot monitor, check the Telegram webhook alert contact.

5. **Verify**: Temporarily stop the trading system on VPS (`sudo systemctl stop trading-system`), wait 5 minutes, confirm Telegram alert arrives. Restart the system, confirm recovery alert arrives.

### What You Get

- Alert within 5 minutes of VPS going down.
- Recovery notification when it comes back.
- Free status page at `https://stats.uptimerobot.com/xxxxx` (optional, shareable).
- Historical uptime data and response time graphs.

### Limitations

- No consecutive-failure gating (alerts on first failure — occasionally a false positive on transient network blip).
- Alert message format is fixed (UptimeRobot variables only).
- Cannot run custom logic (e.g., check portfolio health, not just HTTP 200).

---

## Option B: n8n Workflow (Advanced)

### Why n8n

- Consecutive failure detection (3 strikes before alerting — fewer false positives).
- Custom alert messages with rich context.
- Extensible: add equity checks, trade count monitoring, multi-VPS support later.
- Can be self-hosted or cloud-hosted.

### Cost Analysis

| Hosting | Cost | Execution Limit | 5-min checks/mo | 30-min checks/mo | 60-min checks/mo |
|---------|------|-----------------|------------------|-------------------|-------------------|
| n8n.cloud Free | $0 | 500/month | 8,640 (over) | 1,440 (over) | 720 (over) |
| n8n.cloud Starter | ~$20/mo | 2,500/month | 8,640 (over) | 1,440 (ok-ish) | 720 (ok) |
| Self-hosted ($3 VPS) | ~$3/mo | Unlimited | All ok | All ok | All ok |

**Recommendation**: Self-host n8n on a separate $3/month VPS (e.g., Hetzner CX11, RackNerd, BuyVM). This gives unlimited executions at 5-minute intervals for $3/month. The watchdog VPS must be on a **different provider** than the trading VPS to avoid correlated outages.

### Workflow Design

```
[Cron: every 5 min]
    |
    v
[HTTP Request: GET /api/portfolio, 10s timeout]
    |
    +-- Success --> [Read failure counter from static data]
    |                   |
    |                   +-- Was down? (counter >= 3)
    |                   |       |
    |                   |       YES --> [Send recovery Telegram] --> [Reset counter to 0]
    |                   |       NO  --> [Reset counter to 0]
    |                   |
    +-- Failure --> [Increment failure counter]
                        |
                        +-- Counter >= 3?
                        |       |
                        |       YES --> [Already alerted?]
                        |       |           |
                        |       |           NO --> [Send down Telegram] --> [Set alerted=true]
                        |       |           YES --> (skip, already alerted)
                        |       |
                        |       NO --> (wait for more failures)
```

**State tracking**: n8n's static data (per-workflow persistent key-value store) holds:
- `failureCount` (number): consecutive failures
- `alerted` (boolean): whether a down alert has been sent

### n8n Workflow JSON

The following JSON can be imported directly into n8n via **Settings > Import from File**.

```json
{
  "name": "Tradebot VPS Watchdog",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "minutes",
              "minutesInterval": 5
            }
          ]
        }
      },
      "id": "cron-trigger",
      "name": "Every 5 Minutes",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1.2,
      "position": [0, 0]
    },
    {
      "parameters": {
        "url": "https://tradebot.codeandcraft.ai/api/portfolio",
        "options": {
          "timeout": 10000,
          "allowUnauthorizedCerts": false
        }
      },
      "id": "http-check",
      "name": "Health Check",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [220, 0],
      "continueOnFail": true
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict"
          },
          "conditions": [
            {
              "id": "check-error",
              "leftValue": "={{ $json.error }}",
              "rightValue": "",
              "operator": {
                "type": "exists",
                "operation": "exists"
              }
            }
          ],
          "combinator": "or"
        }
      },
      "id": "is-failure",
      "name": "Request Failed?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 2.2,
      "position": [440, 0]
    },
    {
      "parameters": {
        "jsCode": "// SUCCESS path: check if we were previously down\nconst staticData = $getWorkflowStaticData('global');\nconst failureCount = staticData.failureCount || 0;\nconst alerted = staticData.alerted || false;\n\n// Reset failure counter\nstaticData.failureCount = 0;\nstaticData.alerted = false;\n\nreturn [{\n  json: {\n    wasDown: alerted,\n    previousFailures: failureCount,\n    status: 'up'\n  }\n}];"
      },
      "id": "success-handler",
      "name": "Handle Success",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [660, -120]
    },
    {
      "parameters": {
        "jsCode": "// FAILURE path: increment counter, check threshold\nconst staticData = $getWorkflowStaticData('global');\nstaticData.failureCount = (staticData.failureCount || 0) + 1;\nconst alerted = staticData.alerted || false;\n\nconst shouldAlert = staticData.failureCount >= 3 && !alerted;\nif (shouldAlert) {\n  staticData.alerted = true;\n}\n\nreturn [{\n  json: {\n    failureCount: staticData.failureCount,\n    shouldAlert: shouldAlert,\n    alreadyAlerted: alerted,\n    status: 'down'\n  }\n}];"
      },
      "id": "failure-handler",
      "name": "Handle Failure",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [660, 120]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict"
          },
          "conditions": [
            {
              "id": "was-down",
              "leftValue": "={{ $json.wasDown }}",
              "rightValue": true,
              "operator": {
                "type": "boolean",
                "operation": "equals"
              }
            }
          ],
          "combinator": "and"
        }
      },
      "id": "was-down-check",
      "name": "Was Previously Down?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 2.2,
      "position": [880, -120]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict"
          },
          "conditions": [
            {
              "id": "should-alert",
              "leftValue": "={{ $json.shouldAlert }}",
              "rightValue": true,
              "operator": {
                "type": "boolean",
                "operation": "equals"
              }
            }
          ],
          "combinator": "and"
        }
      },
      "id": "should-alert-check",
      "name": "Should Alert?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 2.2,
      "position": [880, 120]
    },
    {
      "parameters": {
        "chatId": "={{ $credentials.telegramChatId }}",
        "text": "=TRADEBOT RECOVERED\n\nTradebot Quant is back online.\nURL: https://tradebot.codeandcraft.ai\nDowntime failures: {{ $json.previousFailures }}",
        "additionalFields": {
          "parse_mode": "Markdown"
        }
      },
      "id": "recovery-alert",
      "name": "Send Recovery Alert",
      "type": "n8n-nodes-base.telegram",
      "typeVersion": 1.2,
      "position": [1100, -180],
      "credentials": {
        "telegramApi": {
          "id": "CONFIGURE_ME",
          "name": "Tradebot Telegram"
        }
      }
    },
    {
      "parameters": {
        "chatId": "={{ $credentials.telegramChatId }}",
        "text": "=TRADEBOT DOWN\n\nTradebot Quant is unreachable!\nURL: https://tradebot.codeandcraft.ai\nConsecutive failures: {{ $json.failureCount }}\n\nCheck VPS: ssh trader@187.77.132.195\nRestart: sudo systemctl restart trading-system",
        "additionalFields": {
          "parse_mode": "Markdown"
        }
      },
      "id": "down-alert",
      "name": "Send Down Alert",
      "type": "n8n-nodes-base.telegram",
      "typeVersion": 1.2,
      "position": [1100, 60],
      "credentials": {
        "telegramApi": {
          "id": "CONFIGURE_ME",
          "name": "Tradebot Telegram"
        }
      }
    }
  ],
  "connections": {
    "Every 5 Minutes": {
      "main": [
        [{ "node": "Health Check", "type": "main", "index": 0 }]
      ]
    },
    "Health Check": {
      "main": [
        [{ "node": "Request Failed?", "type": "main", "index": 0 }]
      ]
    },
    "Request Failed?": {
      "main": [
        [{ "node": "Handle Failure", "type": "main", "index": 0 }],
        [{ "node": "Handle Success", "type": "main", "index": 0 }]
      ]
    },
    "Handle Success": {
      "main": [
        [{ "node": "Was Previously Down?", "type": "main", "index": 0 }]
      ]
    },
    "Handle Failure": {
      "main": [
        [{ "node": "Should Alert?", "type": "main", "index": 0 }]
      ]
    },
    "Was Previously Down?": {
      "main": [
        [{ "node": "Send Recovery Alert", "type": "main", "index": 0 }],
        []
      ]
    },
    "Should Alert?": {
      "main": [
        [{ "node": "Send Down Alert", "type": "main", "index": 0 }],
        []
      ]
    }
  },
  "settings": {
    "executionOrder": "v1"
  },
  "tags": [
    { "name": "monitoring" },
    { "name": "tradebot" }
  ]
}
```

### n8n Setup Instructions

#### Option 1: Self-Hosted (Recommended for Unlimited Checks)

1. **Provision a separate VPS** ($3/month, different provider than trading VPS):
   - Hetzner CX11 (1 vCPU, 2GB RAM) — EUR 3.29/mo
   - RackNerd 1GB KVM — ~$1.50/mo on deals
   - BuyVM Slice 512 — $2/mo

2. **Install n8n**:
   ```bash
   # Install Node.js 20
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs

   # Install n8n globally
   sudo npm install -g n8n

   # Create systemd service
   sudo tee /etc/systemd/system/n8n.service > /dev/null <<'EOF'
   [Unit]
   Description=n8n workflow automation
   After=network.target

   [Service]
   Type=simple
   User=n8n
   Environment=N8N_PORT=5678
   Environment=N8N_PROTOCOL=http
   Environment=GENERIC_TIMEZONE=Asia/Singapore
   ExecStart=/usr/bin/n8n start
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   EOF

   sudo useradd -r -s /bin/false n8n
   sudo systemctl enable --now n8n
   ```

3. **Access n8n**: Open `http://<n8n-vps-ip>:5678` in browser, create admin account.

4. **Configure Telegram credentials**:
   - Go to **Settings** > **Credentials** > **Add Credential** > **Telegram API**
   - Name: `Tradebot Telegram`
   - Access Token: `<TELEGRAM_BOT_TOKEN>` (from trading VPS `.env`)
   - Note: n8n's built-in Telegram node uses the bot token credential. The `chatId` is specified per-node in the workflow. After importing, update both Telegram nodes' `chatId` fields with `<TELEGRAM_CHAT_ID>` from the trading VPS `.env`.

5. **Import workflow**:
   - Go to **Workflows** > **Import from File**
   - Paste the JSON above (or save as `.json` and upload)
   - Update both Telegram nodes to use the `Tradebot Telegram` credential
   - Update `chatId` in both Telegram nodes with actual chat ID
   - **Activate** the workflow (toggle in top-right)

6. **Test**:
   - Click **Execute Workflow** manually — should succeed (green path)
   - Stop trading system on VPS, click Execute 3 times — third time should send down alert
   - Start trading system, click Execute — should send recovery alert
   - Activate the workflow for automatic 5-minute scheduling

#### Option 2: n8n.cloud

Not recommended due to execution limits. The free tier (500 executions/month) only supports one check every ~90 minutes. The Starter plan ($20/month) supports ~every 30 minutes. Neither is cost-effective compared to a $3 self-hosted VPS with unlimited executions.

If you still want n8n.cloud:
1. Sign up at https://n8n.cloud
2. Follow steps 4-6 above (same workflow import process)
3. Change cron interval from 5 minutes to 60 minutes to stay under limits

### n8n Credential Security

- Telegram bot token is stored in n8n's encrypted credential store, never in workflow JSON.
- The `chatId` field in Telegram nodes contains the chat ID. For additional security, this can also be stored as an n8n credential variable and referenced as `={{ $vars.TELEGRAM_CHAT_ID }}`.
- The workflow JSON above uses `CONFIGURE_ME` as credential ID placeholder — must be replaced during import.

---

## Comparison

| Criteria | UptimeRobot (Free) | n8n (Self-Hosted) |
|----------|---------------------|-------------------|
| Cost | $0/month | ~$3/month (VPS) |
| Setup time | 5 minutes | 30 minutes |
| Check interval | 5 minutes | 5 minutes (configurable) |
| Consecutive failure gating | No (alerts on first failure) | Yes (3 strikes) |
| False positive rate | Low (occasional) | Very low |
| Recovery alerts | Yes | Yes |
| Custom alert messages | Limited | Full control |
| Status page | Free included | No (needs extra setup) |
| Maintenance | None | VPS updates, n8n updates |
| Extensibility | None | Full workflow engine |
| Multi-VPS monitoring | Yes (50 monitors) | Yes (add HTTP nodes) |
| Infrastructure risk | UptimeRobot goes down (rare) | Watchdog VPS goes down |

## Implementation Plan

### Phase 1 (Now): UptimeRobot

1. Create UptimeRobot account — 5 minutes
2. Add HTTP monitor for `https://tradebot.codeandcraft.ai/api/portfolio` — 2 minutes
3. Configure Telegram webhook alert contact — 5 minutes
4. Test by stopping/starting trading system — 10 minutes
5. Done. Total: ~20 minutes, $0/month.

### Phase 2 (Later, Optional): n8n

Only pursue if:
- You want consecutive failure detection (fewer false positives)
- You plan to build other automations (e.g., daily equity reports, multi-instance monitoring for TradeHive)
- You want full control over alert content and logic

## Health Endpoint Details

**Endpoint**: `GET https://tradebot.codeandcraft.ai/api/portfolio`
- **Auth**: None required (public endpoint per nginx config)
- **Expected response**: HTTP 200 with JSON body containing portfolio state
- **Failure modes**: HTTP 5xx, connection timeout, connection refused, SSL error
- **Alternative endpoints**: `/api/market` (also public) — use as secondary check if needed
