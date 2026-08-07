/*
 * MemConflict conversation browser — Alpine.js component.
 *
 * Runs entirely from file:// — no fetch, no modules, no build step.
 * window.MEMCONFLICT_DATA (defined by data.js, loaded before this script)
 * is an array of "persona" objects; see the spec in the project README
 * for the full shape. This file only reads that global, never mutates it.
 */

(function () {
  'use strict';

  // ------------------------------------------------------------------
  // Small standalone helpers (no Alpine/this dependency, easy to test)
  // ------------------------------------------------------------------

  // "Career_Status" -> "Career Status"
  function humanizeKey(key) {
    return String(key).replace(/_/g, ' ');
  }

  // Turn a nested plain-object/array structure into a flat list of rows
  // the template can x-for over, e.g.:
  //   { Career_Status: { Job_Title: "Senior" } }
  // becomes
  //   [ {depth:0, key:"Career Status", value:"", isLeaf:false},
  //     {depth:1, key:"Job Title", value:"Senior", isLeaf:true} ]
  function flattenObject(obj, depth) {
    depth = depth || 0;
    var rows = [];

    if (obj === null || obj === undefined) {
      return rows;
    }

    if (Array.isArray(obj)) {
      obj.forEach(function (item, i) {
        var label = '#' + (i + 1);
        if (item !== null && typeof item === 'object') {
          rows.push({ depth: depth, key: label, value: '', isLeaf: false });
          rows = rows.concat(flattenObject(item, depth + 1));
        } else {
          rows.push({ depth: depth, key: label, value: leafToString(item), isLeaf: true });
        }
      });
      return rows;
    }

    if (typeof obj === 'object') {
      Object.keys(obj).forEach(function (k) {
        var v = obj[k];
        var label = humanizeKey(k);
        if (v !== null && typeof v === 'object' && Object.keys(v).length > 0) {
          rows.push({ depth: depth, key: label, value: '', isLeaf: false });
          rows = rows.concat(flattenObject(v, depth + 1));
        } else if (v !== null && typeof v === 'object') {
          // empty object/array — show as a leaf with an empty marker
          rows.push({ depth: depth, key: label, value: '—', isLeaf: true });
        } else {
          rows.push({ depth: depth, key: label, value: leafToString(v), isLeaf: true });
        }
      });
      return rows;
    }

    // A primitive was passed directly (rare, but keep it safe).
    rows.push({ depth: depth, key: '', value: leafToString(obj), isLeaf: true });
    return rows;
  }

  function leafToString(v) {
    if (v === null || v === undefined || v === '') {
      return '—';
    }
    return String(v);
  }

  var SESSION_TYPE_LABELS = {
    initial_reveal: 'Initial Reveal',
    update: 'Update',
    chitchat: 'Chitchat',
    future_plan: 'Future Plan'
  };

  var SESSION_TYPE_CLASSES = {
    initial_reveal: 'badge-initial_reveal',
    update: 'badge-update',
    chitchat: 'badge-chitchat',
    future_plan: 'badge-future_plan'
  };

  // ------------------------------------------------------------------
  // Alpine component
  // ------------------------------------------------------------------

  function app() {
    return {
      // ---- state ----
      personas: [],
      personaSearch: '',
      currentPersonaIdx: 0,
      currentSessionIdx: null,
      showDetails: false,
      detailsTab: 'profile',

      // ---- lifecycle ----
      init: function () {
        var self = this;
        this.loadData();
        // The file:// build has window.MEMCONFLICT_DATA ready at init time.
        // The artifact build inflates it asynchronously (gzip decompress) and
        // dispatches this event when ready; re-run loadData() then. Harmless
        // for the file:// build, where the event never fires.
        window.addEventListener('memconflict-data-ready', function () {
          self.loadData();
        });
      },

      loadData: function () {
        var data = window.MEMCONFLICT_DATA;
        if (Array.isArray(data)) {
          this.personas = data;
        } else {
          // Keep `personas` an array-like with an `error` flag the
          // template checks via `personas.error`.
          this.personas = [];
          this.personas.error = true;
        }
        // Auto-select the first persona, if any.
        if (this.personas.length > 0) {
          this.currentPersonaIdx = this.personas[0].idx;
        }
      },

      // ---- derived data ----
      filteredPersonas: function () {
        var list = this.personas || [];
        var q = (this.personaSearch || '').trim().toLowerCase();
        if (!q) {
          return list;
        }
        return list.filter(function (p) {
          var name = (p.name || '').toLowerCase();
          var seed = (p.persona_seed || '').toLowerCase();
          return name.indexOf(q) !== -1 || seed.indexOf(q) !== -1;
        });
      },

      currentPersona: function () {
        if (!this.personas || this.personas.length === 0) {
          return null;
        }
        var found = null;
        for (var i = 0; i < this.personas.length; i++) {
          if (this.personas[i].idx === this.currentPersonaIdx) {
            found = this.personas[i];
            break;
          }
        }
        return found;
      },

      currentSessions: function () {
        var p = this.currentPersona();
        return (p && Array.isArray(p.sessions)) ? p.sessions : [];
      },

      currentSession: function () {
        if (this.currentSessionIdx === null || this.currentSessionIdx === undefined) {
          return null;
        }
        var sessions = this.currentSessions();
        return sessions[this.currentSessionIdx] || null;
      },

      // ---- actions ----
      selectPersona: function (idx) {
        this.currentPersonaIdx = idx;
        this.currentSessionIdx = null;
        // detailsTab is intentionally left as-is per spec.
      },

      selectSession: function (idx) {
        this.currentSessionIdx = idx;
        if (this.showDetails) {
          this.detailsTab = 'session';
        }
      },

      toggleDetails: function () {
        this.showDetails = !this.showDetails;
      },

      // ---- display helpers ----
      sessionTypeLabel: function (type) {
        return SESSION_TYPE_LABELS[type] || humanizeKey(type || '');
      },

      sessionTypeClass: function (type) {
        return SESSION_TYPE_CLASSES[type] || 'badge-chitchat';
      },

      messageCount: function (session) {
        return (session && Array.isArray(session.messages)) ? session.messages.length : 0;
      },

      hasAnnotations: function (session) {
        if (!session) {
          return false;
        }
        var count =
          (session.static_conflicts ? session.static_conflicts.length : 0) +
          (session.conditional_conflicts ? session.conditional_conflicts.length : 0) +
          (session.session_questions ? session.session_questions.length : 0);
        return count > 0;
      },

      // ---- recursive object -> rows renderer (used across details panel) ----
      flattenObject: function (obj) {
        return flattenObject(obj, 0);
      }
    };
  }

  // Expose globally in case it's ever needed directly (e.g. console debugging);
  // the canonical registration is via Alpine.data below.
  window.app = app;

  document.addEventListener('alpine:init', function () {
    Alpine.data('app', app);
  });
})();
