/**
 * satellite.js — Cesium satellite entity, orbit trail, and anomaly state.
 *
 * Requires:  window.cesiumViewer  (set by index.html before this script loads)
 * Exports:   window.satellite     (public API consumed by telemetry.js)
 */
(function (global) {
  'use strict';

  var viewer = global.cesiumViewer;
  if (!viewer) {
    console.error('[satellite.js] window.cesiumViewer not found — ensure this'
      + ' script is loaded after the Cesium viewer is created in index.html.');
    return;
  }

  // ── Internal state ────────────────────────────────────────────────────────
  var currentAlertLevel = 'NORMAL';
  var satEntity         = null;
  var markerEntity      = null;   // always-visible point (orbital-distance)
  var pulseEntity       = null;   // CRITICAL pulsing red ring
  var pulseInterval     = null;
  var trailPositions    = [];     // Cartesian3 ring-buffer, max MAX_TRAIL points
  var trailEntityBack   = null;   // oldest third — opacity 0.25
  var trailEntityMid    = null;   // middle third  — opacity 0.55
  var trailEntityFront  = null;   // newest third  — opacity 1.00
  var trailColor        = Cesium.Color.DODGERBLUE;
  var MAX_TRAIL         = 90;

  // ── Colour helpers ────────────────────────────────────────────────────────
  function getSatelliteColor(alertLevel) {
    switch ((alertLevel || '').toUpperCase()) {
      case 'CRITICAL': return Cesium.Color.RED;
      case 'WARNING':  return Cesium.Color.ORANGE;
      default:         return Cesium.Color.DODGERBLUE;
    }
  }

  function getBodyMaterial(alertLevel) {
    switch ((alertLevel || '').toUpperCase()) {
      case 'CRITICAL': return Cesium.Color.RED.withAlpha(0.88);
      case 'WARNING':  return Cesium.Color.ORANGE.withAlpha(0.88);
      default:         return Cesium.Color.fromCssColorString('#4FC3F7').withAlpha(0.88);
    }
  }

  // ── Trail segment builder ─────────────────────────────────────────────────
  // Three overlapping polylines (oldest→newest) with increasing opacity give
  // the visual fade effect without per-vertex colour support.
  function makeTrailSegment(sliceFn, opacityFn) {
    return viewer.entities.add({
      polyline: {
        positions: new Cesium.CallbackProperty(function () {
          return trailPositions.length >= 2 ? sliceFn(trailPositions) : [];
        }, false),
        width: 1.5,
        material: new Cesium.ColorMaterialProperty(
          new Cesium.CallbackProperty(function () {
            return trailColor.withAlpha(opacityFn());
          }, false)
        ),
        clampToGround: false,
        arcType: Cesium.ArcType.NONE,
      },
    });
  }

  trailEntityBack  = makeTrailSegment(
    function (p) { var t = Math.floor(p.length / 3); return p.slice(0, t + 1); },
    function () { return 0.25; }
  );
  trailEntityMid   = makeTrailSegment(
    function (p) { var t = Math.floor(p.length / 3); return p.slice(t, 2 * t + 1); },
    function () { return 0.55; }
  );
  trailEntityFront = makeTrailSegment(
    function (p) { var t = Math.floor(p.length / 3); return p.slice(2 * t); },
    function () { return 1.00; }
  );

  // ── Satellite box entity ──────────────────────────────────────────────────
  // Box dimensions represent a 2U CubeSat scaled for visibility.
  // The companion point (markerEntity) ensures visibility at orbital scale.
  satEntity = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(0, 0, 550000),
    box: {
      // 2 m × 1 m × 0.5 m CubeSat form-factor (×1 scale; zoom in to see detail)
      dimensions: new Cesium.Cartesian3(2.0, 1.0, 0.5),
      material:   getBodyMaterial('NORMAL'),
      outline:    true,
      outlineColor: Cesium.Color.WHITE.withAlpha(0.5),
    },
    label: {
      text:              'SAT-01',
      font:              '13px "Courier New", monospace',
      fillColor:         Cesium.Color.WHITE,
      outlineColor:      Cesium.Color.BLACK,
      outlineWidth:      2,
      style:             Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset:       new Cesium.Cartesian2(0, -16),
      horizontalOrigin:  Cesium.HorizontalOrigin.CENTER,
      verticalOrigin:    Cesium.VerticalOrigin.BOTTOM,
      // Fade label when camera is very far out
      scaleByDistance:       new Cesium.NearFarScalar(5e5, 1.2, 3e7, 0.4),
      translucencyByDistance: new Cesium.NearFarScalar(2e7, 1.0, 4e7, 0.0),
    },
  });

  // Always-visible orbital-scale point marker (fixed pixel size)
  markerEntity = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(0, 0, 550000),
    point: {
      pixelSize:    10,
      color:        Cesium.Color.fromCssColorString('#4FC3F7').withAlpha(0.9),
      outlineColor: Cesium.Color.WHITE.withAlpha(0.4),
      outlineWidth: 1,
      // Reduce size when zoomed in (the box takes over)
      scaleByDistance: new Cesium.NearFarScalar(5e5, 0.5, 3e7, 1.0),
    },
  });

  // ── Helper: sync position of all position-carrying entities ──────────────
  function _setAllPositions(cartesian) {
    if (satEntity)    satEntity.position    = new Cesium.ConstantPositionProperty(cartesian);
    if (markerEntity) markerEntity.position = new Cesium.ConstantPositionProperty(cartesian);
    if (pulseEntity)  pulseEntity.position  = new Cesium.ConstantPositionProperty(cartesian);
  }

  // ── Public API ────────────────────────────────────────────────────────────
  global.satellite = {

    getSatelliteColor: getSatelliteColor,

    /**
     * Move satellite to new orbital position.
     * @param {number} lat    – geodetic latitude  (degrees)
     * @param {number} lon    – geodetic longitude (degrees)
     * @param {number} altKm  – altitude above WGS-84 ellipsoid (km)
     */
    updateSatellitePosition: function (lat, lon, altKm) {
      var pos = Cesium.Cartesian3.fromDegrees(lon, lat, altKm * 1000);
      _setAllPositions(pos);
      viewer.scene.requestRender();
    },

    /**
     * Push a new point onto the orbit trail (ring-buffer, max 90 points).
     */
    addOrbitTrailPoint: function (lat, lon, altKm) {
      trailPositions.push(Cesium.Cartesian3.fromDegrees(lon, lat, altKm * 1000));
      if (trailPositions.length > MAX_TRAIL) trailPositions.shift();
      viewer.scene.requestRender();
    },

    /** Remove all trail points. */
    clearOrbitTrail: function () {
      trailPositions.length = 0;
      viewer.scene.requestRender();
    },

    /**
     * Update satellite colour, trail colour, pulsing ring, and camera.
     * @param {string} alertLevel – 'NORMAL' | 'WARNING' | 'CRITICAL'
     * @param {string} anomalyType
     */
    updateAnomalyState: function (alertLevel, anomalyType) {
      var level = (alertLevel || 'NORMAL').toUpperCase();
      var stateChanged = (level !== currentAlertLevel);
      currentAlertLevel = level;

      // ── Satellite body colour ──────────────────────────────────────────
      if (satEntity) satEntity.box.material = getBodyMaterial(level);

      // ── Marker point colour ───────────────────────────────────────────
      if (markerEntity) {
        markerEntity.point.color = getBodyMaterial(level);
      }

      // ── Trail colour ──────────────────────────────────────────────────
      trailColor = getSatelliteColor(level);

      // ── CRITICAL pulsing ring ─────────────────────────────────────────
      if (level === 'CRITICAL') {
        if (!pulseEntity) {
          var curPos = satEntity && satEntity.position
            ? satEntity.position.getValue(Cesium.JulianDate.now())
            : Cesium.Cartesian3.fromDegrees(0, 0, 550000);

          pulseEntity = viewer.entities.add({
            position: curPos,
            point: {
              pixelSize:    12,
              color:        Cesium.Color.RED.withAlpha(0.85),
              outlineColor: Cesium.Color.RED,
              outlineWidth: 2,
            },
          });
        }
        if (!pulseInterval) {
          var big = false;
          pulseInterval = setInterval(function () {
            if (pulseEntity) {
              big = !big;
              pulseEntity.point.pixelSize = big ? 20 : 12;
              viewer.scene.requestRender();
            }
          }, 500);
        }
      } else {
        // Clear pulse when leaving CRITICAL
        if (pulseInterval) { clearInterval(pulseInterval); pulseInterval = null; }
        if (pulseEntity)   { viewer.entities.remove(pulseEntity); pulseEntity = null; }
      }

      // ── Camera flyTo on state change ──────────────────────────────────
      if (stateChanged && satEntity && satEntity.position) {
        var pos3 = satEntity.position.getValue(Cesium.JulianDate.now());
        if (pos3) {
          var carto   = Cesium.Cartographic.fromCartesian(pos3);
          var flyLon  = Cesium.Math.toDegrees(carto.longitude);
          var flyLat  = Cesium.Math.toDegrees(carto.latitude);
          var flyAlt  = carto.height;
          viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(flyLon, flyLat, flyAlt + 500000),
            duration: 2.0,
          });
        }
      }

      viewer.scene.requestRender();
    },
  };

})(window);
