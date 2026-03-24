// OpenFang Traces Page — Distributed tracing viewer
'use strict';

function tracesPage() {
  return {
    traces: [],
    total: 0,
    selectedTrace: null,
    spans: [],
    loading: true,
    detailLoading: false,
    filterAgent: '',
    filterTrigger: '',
    expandedSpans: {},

    async loadData() {
      return this.loadTraces();
    },

    async loadTraces() {
      this.loading = true;
      try {
        var params = new URLSearchParams();
        params.set('limit', '50');
        if (this.filterAgent) params.set('agent', this.filterAgent);
        if (this.filterTrigger) params.set('trigger', this.filterTrigger);
        var data = await OpenFangAPI.get('/api/traces?' + params.toString());
        this.traces = data.traces || [];
        this.total = data.total || 0;
      } catch (e) {
        this.traces = [];
        this.total = 0;
      }
      this.loading = false;
    },

    async selectTrace(traceId) {
      this.detailLoading = true;
      this.expandedSpans = {};
      try {
        var data = await OpenFangAPI.get('/api/traces/' + traceId);
        this.selectedTrace = data.trace;
        this.spans = data.spans || [];
      } catch (e) {
        this.selectedTrace = null;
        this.spans = [];
      }
      this.detailLoading = false;
    },

    closeDetail() {
      this.selectedTrace = null;
      this.spans = [];
      this.expandedSpans = {};
    },

    toggleSpan(spanId) {
      if (this.expandedSpans[spanId]) {
        delete this.expandedSpans[spanId];
      } else {
        this.expandedSpans[spanId] = true;
      }
    },

    isExpanded(spanId) {
      return !!this.expandedSpans[spanId];
    },

    formatTime(iso) {
      if (!iso) return '-';
      try {
        var d = new Date(iso);
        // Beijing time (UTC+8)
        var bj = new Date(d.getTime() + 8 * 3600 * 1000);
        var hh = String(bj.getUTCHours()).padStart(2, '0');
        var mm = String(bj.getUTCMinutes()).padStart(2, '0');
        var ss = String(bj.getUTCSeconds()).padStart(2, '0');
        var MM = String(bj.getUTCMonth() + 1).padStart(2, '0');
        var DD = String(bj.getUTCDate()).padStart(2, '0');
        return MM + '/' + DD + ' ' + hh + ':' + mm + ':' + ss;
      } catch (e) {
        return iso;
      }
    },

    formatDuration(ms) {
      if (ms == null) return '-';
      if (ms < 1000) return ms + 'ms';
      if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
      return (ms / 60000).toFixed(1) + 'm';
    },

    formatTokens(n) {
      if (n == null || n === 0) return '-';
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return String(n);
    },

    triggerColor(type) {
      var map = { user: '#4f46e5', tick: '#059669', cron: '#d97706', plan: '#7c3aed', webhook: '#dc2626' };
      return map[type] || '#6b7280';
    },

    kindColor(kind) {
      var map = { hook: '#3b82f6', llm: '#8b5cf6', tool: '#10b981', custom: '#6b7280' };
      return map[kind] || '#6b7280';
    },

    statusColor(status) {
      if (status === 'completed') return '#059669';
      if (status === 'error') return '#dc2626';
      return '#d97706';
    },

    get uniqueAgents() {
      var seen = {};
      this.traces.forEach(function(t) { if (t.agent_name) seen[t.agent_name] = true; });
      return Object.keys(seen).sort();
    },

    async doCleanup() {
      try {
        var data = await OpenFangAPI.delete('/api/traces/cleanup');
        OpenFangToast.success('Cleaned up ' + (data.removed || 0) + ' trace(s)');
        this.loadTraces();
      } catch (e) {
        OpenFangToast.error('Cleanup failed: ' + e.message);
      }
    }
  };
}
