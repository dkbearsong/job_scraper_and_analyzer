# Hiring.cafe Cloudflare Bypass Setup Guide

This document explains the Cloudflare bypass features implemented in `app/scrapers/hiring_cafe_adapter.py` and how to configure them.

## Overview

The updated adapter includes multiple layers of anti-detection:

1. **Stealth Automation** - Patches Playwright's automation fingerprint
2. **Fingerprint Randomization** - Rotates user agents, viewports, and locales
3. **Proxy Rotation** - Distributes requests across multiple IPs
4. **CAPTCHA Solving** - Automatic solving via 2Captcha or CapSolver
5. **Cloudflare Detection** - Identifies and handles challenge pages

## Configuration

### Environment Variables (.env)

Add these to your `.env` file:

```bash
# Proxy List (comma-separated)
# HIRING_CAFE_PROXY_LIST="http://proxy1:8080,http://user:pass@proxy2:8080"
HIRING_CAFE_PROXY_LIST=""

# CAPTCHA Service
CAPTCHA_SERVICE="2captcha"  # or "capsolver"

# API Keys (choose based on service)
# TWOCAPTCHA_API_KEY="your_key_here"
# CAPSOLVER_API_KEY="your_key_here"
# CAPTCHA_API_KEY="your_key_here"
```

### Scrapers Config (scrapers_config.yaml)

```yaml
scrapers:
  - name: hiring_cafe
    enabled: true
    adapter: hiring_cafe_adapter
    config:
      max_pages: 5
      headless: false
      urls_csv: "documents/hiring_cafe_urls.csv"
      
      # Optional: Proxy list (overrides .env)
      proxy_list:
        - "http://proxy1.example.com:8080"
        - "http://user:pass@proxy2.example.com:8080"
      
      # Optional: CAPTCHA solving
      captcha_api_key: "your_api_key"
      captcha_service: "2captcha"  # or "capsolver"
```

## Feature Details

### 1. Stealth Scripts

The adapter injects JavaScript to mask automation:

- **navigator.webdriver** - Patched to return `false`
- **Chrome object** - Added fake `runtime` property
- **Permissions API** - Intercepted to return realistic responses
- **Plugins** - Mocked with realistic count
- **Languages** - Set to `['en-US', 'en']`
- **Element prototypes** - Preserved `scroll`, `getBoundingClientRect`, etc.

### 2. Fingerprint Randomization

Each scrape session gets a random:

- **User Agent** - Pool of 8 realistic Chrome/Firefox/Edge agents
- **Viewport** - Common sizes: 1280x720, 1366x768, 1440x900, etc.
- **Locale** - en-US, en-GB, en-CA, en-AU
- **Timezone** - Matched to locale (America/New_York, Europe/London, etc.)

New browser context created per URL with unique fingerprint.

### 3. Proxy Rotation

Supports residential and datacenter proxies:

```bash
# Single proxy
HIRING_CAFE_PROXY_LIST="http://proxy.example.com:8080"

# Multiple proxies with rotation
HIRING_CAFE_PROXY_LIST="http://proxy1:8080,http://proxy2:8080,http://proxy3:8080"

# With authentication
HIRING_CAFE_PROXY_LIST="http://user:pass@proxy.example.com:8080"
```

**Recommendation:** Use residential proxies for best results. Datacenter IPs are frequently flagged.

### 4. CAPTCHA Solving

#### 2Captcha

Sign up at [2captcha.com](https://2captcha.com), configure:

```bash
CAPTCHA_SERVICE="2captcha"
TWOCAPTCHA_API_KEY="your_key_here"
```

#### CapSolver

Sign up at [capsolver.com](https://capsolver.com), configure:

```bash
CAPTCHA_SERVICE="capsolver"
CAPSOLVER_API_KEY="your_key_here"
```

**How it works:**
1. Detects CAPTCHA (Turnstile, hCaptcha, reCAPTCHA)
2. Extracts site key from page
3. Submits to solving service
4. Injects solution token into page
5. Retries if challenge persists

### 5. Cloudflare Detection & Handling

Automatically detects:

- Page content: "Just a moment", "Checking your browser", "Verify you are human"
- URL patterns: `challenges`, `cloudflare`, `hcaptcha`, `turnstile`
- DOM elements: `[data-turnstile]`, `[data-site-key]`, iframes

Bypass attempts in order:

1. **Auto-bypass** - Wait 5s for stealth scripts to work
2. **CAPTCHA solve** - If API key configured
3. **Turnstile click** - Automate checkbox click
4. **Page reload** - Fresh attempt with new session

## Recommended Setup

### Minimal (No Proxies/CAPTCHA)

```bash
HIRING_CAFE_PROXY_LIST=""
CAPTCHA_SERVICE="2captcha"
# No API keys set
```

Works for low-volume scraping. May hit Cloudflare blocks.

### Standard (With Proxies)

```bash
HIRING_CAFE_PROXY_LIST="http://residential-proxy.example.com:8080"
CAPTCHA_SERVICE="2captcha"
TWOCAPTCHA_API_KEY="your_key"
```

Best balance of cost and reliability.

### Advanced (Full Bypass)

```bash
HIRING_CAFE_PROXY_LIST="http://user:pass@residential1:8080,http://user:pass@residential2:8080"
CAPTCHA_SERVICE="capsolver"
CAPSOLVER_API_KEY="your_key"
```

Maximum evasion with proxy rotation + CAPTCHA solving.

## Testing

Run standalone test:

```bash
python app/scrapers/hiring_cafe_adapter.py --pages 1 --debug
```

Flags:
- `--pages N` - Number of pages to scrape
- `--headless` - Run headless (default)
- `--visible` - Show browser window
- `--debug` - Verbose logging
- `--format summary|json|flat` - Output format

## Troubleshooting

### Cloudflare Still Appears

1. **Add proxies** - Your IP may be flagged
2. **Enable CAPTCHA solving** - Get 2Captcha/CapSolver API key
3. **Rotate user agents** - Already automatic, but may need more variety
4. **Use residential proxies** - Datacenter IPs are heavily blocked

### "All click strategies failed"

- Site may have changed DOM structure
- Check logs for `No multi-listing toggle control found`
- May need to update CSS selectors in `_extract_card_info`

### CAPTCHA Not Solving

- Verify API key is valid (check service dashboard)
- Ensure `captcha_service` matches your provider
- Check logs for "CAPTCHA submission failed"
- Solution tokens expire quickly - increase timeout in `_solve_2captcha`

## Cost Estimates

| Component | Service | Cost |
|-----------|---------|------|
| 2Captcha | Turnstile | $2.99/1000 solves |
| CapSolver | Turnstile | $0.8/1000 solves |
| Residential Proxy | BrightData | ~$5/GB |
| Residential Proxy | SmartProxy | ~$2.5/GB |

For 1000 pages with 10 CAPTCHAs:
- 2Captcha: ~$0.03 + proxy bandwidth
- CapSolver: ~$0.008 + proxy bandwidth

## Limitations

- Cloudflare actively updates detection methods
- CAPTCHA solving adds 10-30s delay per challenge
- Proxy quality varies - test before bulk scraping
- Some challenges require manual intervention

## Security

- Never commit API keys to version control
- Use environment variables or secret managers
- Rotate proxy credentials regularly
- Monitor API usage for abuse