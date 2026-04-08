// OpenFang Characters Page — Live character status cards + wardrobe browser
'use strict';

function charactersPage() {
  return {
    characters: [],
    owner: null,
    arrangements: [],
    knownPlaceKeys: [],
    locationAliases: {},
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
        var allChars = data.characters || [];
        this.characters = allChars.filter(function(c) { return c.is_agent; });
        this.owner = allChars.find(function(c) { return c.is_owner; }) || null;
        this.arrangements = data.arrangements || [];
        this.knownPlaceKeys = data.known_place_keys || [];
        this.locationAliases = data.location_aliases || {};
      } catch(e) {
        this.characters = [];
        this.owner = null;
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

    getLocation(c) {
      return (c.world_state && c.world_state.location) || (c.life_state && c.life_state.location) || '-';
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

    getOwnerLocation() {
      if (!this.owner) return '-';
      return (this.owner.world_state && this.owner.world_state.location) || '-';
    },

    // Location resolution — mirrors backend location_resolver.js (layers 1-3)
    resolveLocation(loc) {
      if (!loc) return null;
      var keys = this.knownPlaceKeys;
      // Layer 1: exact match
      if (keys.indexOf(loc) >= 0) return loc;
      // Layer 2: substring match (bidirectional, ≥2 chars)
      for (var i = 0; i < keys.length; i++) {
        var key = keys[i];
        if (key.length < 2 || loc.length < 2) continue;
        if (loc.indexOf(key) >= 0 || key.indexOf(loc) >= 0) return key;
      }
      // Layer 3: alias cache (null = confirmed no match)
      if (this.locationAliases.hasOwnProperty(loc)) return this.locationAliases[loc];
      return undefined; // cache miss
    },

    samePlace(locA, locB) {
      if (!locA || !locB) return false;
      if (locA === locB) return true;
      var keyA = this.resolveLocation(locA);
      var keyB = this.resolveLocation(locB);
      if (keyA && keyB) return keyA === keyB;
      // Fallback: bidirectional substring
      if (locA.length >= 2 && locB.length >= 2) {
        return locA.indexOf(locB) >= 0 || locB.indexOf(locA) >= 0;
      }
      return false;
    },

    getCoPresenceList(c) {
      var selfLoc = (c.world_state && c.world_state.location) || '';
      var result = [];
      var others = this.characters.concat(this.owner ? [this.owner] : []);
      for (var i = 0; i < others.length; i++) {
        var other = others[i];
        if (other.character_id === c.character_id) continue;
        var otherLoc = (other.world_state && other.world_state.location) || '';
        var mode = this.samePlace(selfLoc, otherLoc) ? 'in_person' : 'remote';
        result.push({ name: other.display_name, mode: mode });
      }
      return result;
    },

    getThoughts(c) {
      return (c.world_state && c.world_state.thoughts) || (c.life_state && c.life_state.thoughts) || [];
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

    getArrangements(c) {
      if (!this.arrangements || !this.arrangements.length) return [];
      var charId = c.character_id;
      return this.arrangements.filter(function(a) {
        return (a.status === 'pending' || a.status === 'in_progress') &&
               a.participants && a.participants.indexOf(charId) >= 0;
      });
    },

    arrangementStatusLabel(status) {
      return status === 'pending' ? '准备中' : status === 'in_progress' ? '进行中' : status;
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
