#!/usr/bin/env node
/**
 * I-STAR Config Generator
 * 
 * Run this script after editing istar_fleet_config.json:
 *   node scripts/generate_config_js.js
 * 
 * This converts the JSON config into a browser-compatible JS file
 * that works in both Electron and standard browsers.
 */

const fs = require('fs');
const path = require('path');

const jsonPath = path.join(__dirname, '..', 'istar_fleet_config.json');
const jsPath = path.join(__dirname, '..', 'istar_config.js');

const config = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
const output = `// AUTO-GENERATED from istar_fleet_config.json — DO NOT EDIT DIRECTLY
// To update: edit istar_fleet_config.json then run: node scripts/generate_config_js.js
const ISTAR_FLEET_CONFIG = ${JSON.stringify(config, null, 2)};
`;

fs.writeFileSync(jsPath, output, 'utf8');
console.log(`[I-STAR] istar_config.js regenerated with ${config.satellites.length} satellites.`);
console.log('[I-STAR] Restart the application for changes to take effect.');
