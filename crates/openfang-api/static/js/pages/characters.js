// OpenFang Characters Page — Live character status cards + wardrobe browser
'use strict';

function charactersPage() {
  return {
    characters: [],
    loading: true,
    autoRefresh: true,
    _timer: null,

    // Wardrobe panel state
    wardrobeAgent: null,
    wardrobeAgentName: '',
    wardrobeItems: [],
    wardrobeLoading: false,
    wardrobeCurrentId: null,

    async loadData() {
      this.loading = true;
      await this.refresh();
      this.loading = false;
      var self = this;
      if (this._timer) clearInterval(this._timer);
      this._timer = setInterval(function() {
        if (self.autoRefresh && !self.wardrobeAgent) self.refresh();
      }, 30000);
    },

    async refresh() {
      try {
        var data = await OpenFangAPI.get('/api/characters');
        this.characters = (data.characters || []).filter(function(c) { return c.is_agent; });
      } catch(e) {
        this.characters = [];
      }
    },

    destroy() {
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
    },

    // ── Wardrobe ──

    async openWardrobe(agentId, agentName) {
      this.wardrobeAgent = agentId;
      this.wardrobeAgentName = agentName;
      this.wardrobeLoading = true;
      this.wardrobeItems = [];
      try {
        var data = await OpenFangAPI.get('/api/characters/' + agentId + '/wardrobe');
        this.wardrobeItems = data.items || [];
        this.wardrobeCurrentId = data.current_item_id || null;
      } catch(e) {
        this.wardrobeItems = [];
      }
      this.wardrobeLoading = false;
    },

    closeWardrobe() {
      this.wardrobeAgent = null;
      this.wardrobeItems = [];
    },

    // ── Helpers ──

    avatarUrl(c) {
      if (!c.agent_id) return '';
      return '/api/characters/' + c.agent_id + '/avatar';
    },

    formatTime(iso) {
      if (!iso) return '-';
      try {
        var d = new Date(iso);
        var bj = new Date(d.getTime() + 8 * 3600 * 1000);
        var hh = String(bj.getUTCHours()).padStart(2, '0');
        var mm = String(bj.getUTCMinutes()).padStart(2, '0');
        return hh + ':' + mm;
      } catch(e) { return '-'; }
    },

    formatDateTime(iso) {
      if (!iso) return '-';
      try {
        var d = new Date(iso);
        var bj = new Date(d.getTime() + 8 * 3600 * 1000);
        var MM = String(bj.getUTCMonth() + 1).padStart(2, '0');
        var DD = String(bj.getUTCDate()).padStart(2, '0');
        var hh = String(bj.getUTCHours()).padStart(2, '0');
        var mm = String(bj.getUTCMinutes()).padStart(2, '0');
        return MM + '/' + DD + ' ' + hh + ':' + mm;
      } catch(e) { return '-'; }
    },

    pct(v) {
      if (v == null) return 0;
      return Math.round(Math.min(1, Math.max(0, v)) * 100);
    },

    barColor(v) {
      if (v == null) return '#6b7280';
      var p = Math.min(1, Math.max(0, v));
      if (p < 0.3) return '#10b981';
      if (p < 0.6) return '#f59e0b';
      return '#ef4444';
    },

    getMood(c) {
      return (c.life_state && c.life_state.mood) || (c.world_state && c.world_state.mood) || '-';
    },

    getLocation(c) {
      return c.location_name || (c.life_state && c.life_state.location) || (c.world_state && c.world_state.location) || '-';
    },

    getPhysical(c) {
      return (c.life_state && c.life_state.physical) || (c.world_state && c.world_state.physical) || {};
    },

    getActivity(c) {
      if (!c.life_state) return null;
      if (c.life_state.sleep_state) return 'sleeping';
      if (c.life_state.current_activity && c.life_state.current_activity.activity) {
        return c.life_state.current_activity.activity;
      }
      return null;
    },

    getMode(c) {
      return (c.life_state && c.life_state.interaction_mode) || 'remote';
    },

    getThoughts(c) {
      return (c.life_state && c.life_state.thoughts) || [];
    },

    getEvents(c) {
      var events = (c.life_state && c.life_state.event_log) || [];
      return events.slice(-5).reverse();
    },

    getTensions(c) {
      return (c.life_state && c.life_state.life_tensions) || (c.world_state && c.world_state.life_tensions) || {};
    },

    getScheduleEntries(c) {
      if (!c.schedule || !c.schedule.entries) return [];
      return c.schedule.entries;
    },

    getOutfit(c) {
      if (c.current_outfit && c.current_outfit.name) return c.current_outfit.name;
      return null;
    },

    getDesires(c) {
      if (!c.desire_memories) return [];
      var arr = Array.isArray(c.desire_memories) ? c.desire_memories : [];
      return arr.filter(function(d) { return !d.fulfilled_at; }).slice(-3);
    },

    getSocialCognition(c) {
      return c.social_cognition || [];
    },

    getLastInteraction(c) {
      if (!c.context_cache) return null;
      return c.context_cache.last_user_interaction_at;
    },

    getChannel(c) {
      if (!c.context_cache || !c.context_cache.last_known_channel) return null;
      return c.context_cache.last_known_channel.channel;
    },

    tensionLabel(key) {
      var m = {
        rest_tension: 'rest', food_tension: 'food',
        domestic_tension: 'domestic', social_tension: 'social',
        solitude_tension: 'solitude'
      };
      return m[key] || key.replace(/_tension$/, '');
    },

    tensionColor(v) {
      if (v == null || v <= 0.3) return '#10b981';
      if (v <= 0.6) return '#f59e0b';
      return '#ef4444';
    }
  };
}
