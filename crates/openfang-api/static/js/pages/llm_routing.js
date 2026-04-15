// LLM Routing — Model assignment + API connectivity test
'use strict';

function llmRoutingPage() {
  return {
    tab: 'routing',
    loading: true,
    loadError: '',
    saving: {},
    // Data
    agents: [],
    slots: {},
    providers: {},
    knownModels: [],
    catalogModels: [],
    // Test
    testResults: {},
    testing: {},
    customTestModel: '',
    customTestMessage: '',

    // ── Slot metadata (display names + grouping) ────────────────────────
    slotMeta: {
      'director':            { name: 'Director 导演',        group: 'director' },
      'cascade:present':     { name: '级联 (用户在场)',       group: 'director' },
      'cascade:absent':      { name: '级联 (用户不在场)',     group: 'director' },
      'simulation':          { name: 'Planning 规划',        group: 'simulation' },
      'simulation:light':    { name: 'Judgment 判断',        group: 'simulation' },
      'simulation:schedule': { name: '日程管理',             group: 'simulation' },
      'mood_assessment':     { name: '心情评估',             group: 'auxiliary' },
      'prompt_rewrite':      { name: 'Prompt 改写',          group: 'auxiliary' },
      'location_resolver':   { name: '地点标准化',            group: 'auxiliary' },
      'observable':          { name: '可观察行为过滤',        group: 'auxiliary' },
      'wardrobe_judgment':   { name: '衣橱匹配',            group: 'auxiliary' },
      'vision':              { name: '图片理解 (Vision)',     group: 'auxiliary' },
      'diary:retrieval':     { name: '日记检索',             group: 'diary' },
      'diary:cognition':     { name: '认知事实提取',          group: 'diary' },
      'npc':                 { name: 'NPC 对话生成',         group: 'npc' },
      'npc:cognition':       { name: 'NPC 认知沉淀',        group: 'npc' },
      'compact':             { name: 'Session 压缩',        group: 'system' },
      'filming':             { name: '拍摄系统',             group: 'filming' },
    },
    groups: [
      { id: 'agent',      name: '角色大脑' },
      { id: 'director',   name: '多角色导演' },
      { id: 'simulation', name: '仿真引擎' },
      { id: 'auxiliary',  name: '辅助调用' },
      { id: 'diary',      name: '日记系统' },
      { id: 'npc',        name: 'NPC' },
      { id: 'system',     name: '系统' },
      { id: 'filming',    name: '拍摄系统' },
    ],

    // ── Lifecycle ───────────────────────────────────────────────────────

    async loadData() {
      this.loading = true;
      this.loadError = '';
      try {
        var routing = await OpenFangAPI.get('/api/llm-routing');
        this.agents = routing.agents || [];
        this.slots = routing.slots || {};
        this.providers = routing.providers || {};
        this.knownModels = routing.known_models || [];
      } catch(e) {
        this.loadError = e.message || 'Failed to load';
      }
      this.loading = false;
    },

    destroy() {},

    // ── Computed ─────────────────────────────────────────────────────────

    get allModelOptions() {
      // Only show known_models — the models the user actually has access to
      return this.knownModels.slice().sort();
    },

    agentSlots() {
      return this.agents.map(function(a) {
        return {
          slot: 'agent:' + a.name,
          name: a.name,
          primary: a.model,
          provider: a.provider,
          fallbacks: (a.fallbacks || []).map(function(f) { return f.model; }),
        };
      });
    },

    mcpSlotsByGroup(groupId) {
      var self = this;
      return Object.keys(this.slotMeta).filter(function(k) {
        return self.slotMeta[k].group === groupId;
      }).map(function(k) {
        var s = self.slots[k] || {};
        return { slot: k, name: self.slotMeta[k].name, primary: s.primary || '', fallback: s.fallback || '', fallback2: s.fallback2 || '' };
      });
    },

    // ── Save ─────────────────────────────────────────────────────────────

    async saveSlot(slot, field, value) {
      var key = slot + ':' + field;
      this.saving[key] = true;
      try {
        await OpenFangAPI.put('/api/llm-routing', { slot: slot, field: field, value: value });
        if (typeof OpenFangToast !== 'undefined') OpenFangToast.success(slot + ' ' + field + ' → ' + value);
        await this.loadData();
      } catch(e) {
        if (typeof OpenFangToast !== 'undefined') OpenFangToast.error('Save failed: ' + e.message);
      }
      this.saving[key] = false;
    },

    // ── Provider → model resolution for test ─────────────────────────────

    resolveProvider(modelId) {
      var providers = this.providers;
      var keys = Object.keys(providers);
      for (var i = 0; i < keys.length; i++) {
        var p = providers[keys[i]];
        var prefixes = p.prefixes || [];
        for (var j = 0; j < prefixes.length; j++) {
          if (modelId.indexOf(prefixes[j]) === 0) {
            return {
              name: keys[i],
              base_url: p.base_url || '',
              base_url_env: p.base_url_env || '',
              api_key_env: p.api_key_env || '',
            };
          }
        }
      }
      // Default to vibecoding
      var vibe = providers.vibecoding;
      return vibe ? { name: 'vibecoding', base_url: vibe.base_url, api_key_env: vibe.api_key_env } : { name: 'unknown' };
    },

    // ── Test ─────────────────────────────────────────────────────────────

    async testModel(modelId, message) {
      this.testing[modelId] = true;
      this.testResults[modelId] = null;
      var prov = this.resolveProvider(modelId);
      try {
        var result = await OpenFangAPI.post('/api/llm-routing/test', {
          model: modelId,
          base_url: prov.base_url || '',
          api_key_env: prov.api_key_env || '',
          message: message || 'Say one word.',
        });
        this.testResults[modelId] = result;
        if (result.status === 'ok') {
          if (typeof OpenFangToast !== 'undefined') OpenFangToast.success(modelId + ' OK (' + result.latency_ms + 'ms)');
        } else {
          if (typeof OpenFangToast !== 'undefined') OpenFangToast.error(modelId + ': ' + (result.error || 'Failed').substring(0, 100));
        }
      } catch(e) {
        this.testResults[modelId] = { status: 'error', error: e.message };
        if (typeof OpenFangToast !== 'undefined') OpenFangToast.error('Test failed: ' + e.message);
      }
      this.testing[modelId] = false;
    },

    async testCustom() {
      var m = this.customTestModel.trim();
      if (!m) return;
      await this.testModel(m, this.customTestMessage.trim() || undefined);
    },

    testBadgeClass(modelId) {
      var r = this.testResults[modelId];
      if (!r) return '';
      return r.status === 'ok' ? 'badge-success' : 'badge-error';
    },

    testBadgeText(modelId) {
      var r = this.testResults[modelId];
      if (!r) return '';
      if (r.status === 'ok') return 'OK ' + r.latency_ms + 'ms';
      return 'Error';
    },
  };
}
