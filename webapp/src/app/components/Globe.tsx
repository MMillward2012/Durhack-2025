'use client';

import React, { useRef, useEffect } from 'react';
import * as Cesium from 'cesium';

// Set the Cesium Ion access token (you can get a free one from cesium.com)
Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyYWRhOTBkNi1iYWQwLTQ4YzYtODFiNy03NDJlOWMxMWZiMGMiLCJpZCI6MjM4NDE4LCJpYXQiOjE3MzA1MTc2MTl9.xmkFgpekJKKBo67CzlRoP4ld-4JTKN5q5JE8CHY7hbE';

interface HeatmapData {
  year: number;
  month: number;
  date_string: string;
  lat_range: number[];
  lon_range: number[];
  prob_grid: number[][];
  statistics: {
    mean_probability: number;
    max_probability: number;
    min_probability: number;
    std_probability: number;
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number;
    total_locations: number;
  };
  climate_info: {
    base_sst_mean: number;
    climate_adjustment: number;
    climate_adjusted_sst_mean: number;
    years_since_baseline: number;
  };
  top_locations: Array<{
    lat: number;
    lon: number;
    risk: number;
  }>;
  grid_resolution: {
    lat_points: number;
    lon_points: number;
  };
  generated_at: string;
}

interface GlobeProps {
  data: HeatmapData | null;
}

const getHeatmapColorString = (probability: number, maxProbability: number): string | null => {
  if (!probability || probability <= 0.001) return null;

  // Use ABSOLUTE probability threshold of 5%, not relative to max
  if (probability < 0.05) {
    return null; // Hide anything below 5% absolute probability
  }
  
  // Map absolute probability 5%-100% to color range
  const absoluteRange = Math.min((probability - 0.05) / 0.95, 1); // 5% -> 100% maps to 0 -> 1
  const eased = Math.pow(absoluteRange, 0.6); // Moderate contrast curve
  
  // Adjust color mapping so 65%+ is red
  // 5% = yellow (60°), 65% = red (0°)
  const redThreshold = (0.65 - 0.05) / 0.95; // 65% maps to this point in our range
  let hue;
  if (absoluteRange >= redThreshold) {
    hue = 0; // Pure red for 65%+
  } else {
    // Yellow to orange for 5%-65%
    hue = 60 * (1 - (absoluteRange / redThreshold));
  }
  
  const saturation = 85 + eased * 15; // 85% -> 100% saturation for intensity
  const lightness = 55 - eased * 15; // 55% -> 40% lightness (darker = more intense)
  const alpha = 0.5 + eased * 0.4; // 0.5 -> 0.9 alpha for good visibility

  return `hsla(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%, ${alpha.toFixed(2)})`;
};

const createHeatmapTexture = (
  latRange: number[],
  lonRange: number[],
  probGrid: number[][],
  maxProbability: number
): { url: string; width: number; height: number } | null => {
  const width = 1024;
  const height = 512;

  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = width;
  baseCanvas.height = height;

  const ctx = baseCanvas.getContext('2d');
  if (!ctx) return null;

  ctx.clearRect(0, 0, width, height);

  const cellWidth = width / lonRange.length;
  const cellHeight = height / latRange.length;

  for (let i = 0; i < latRange.length; i++) {
    for (let j = 0; j < lonRange.length; j++) {
      const probability = probGrid[i]?.[j];
      const fill = getHeatmapColorString(probability ?? 0, maxProbability);
      if (!fill) continue;

      const x = Math.floor(j * cellWidth);
      const y = Math.floor((latRange.length - 1 - i) * cellHeight);

      ctx.fillStyle = fill;
      ctx.fillRect(x, y, Math.ceil(cellWidth) + 1, Math.ceil(cellHeight) + 1);
    }
  }

  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const blurCtx = blurCanvas.getContext('2d');

  if (blurCtx) {
    blurCtx.filter = 'blur(6px)';
    blurCtx.drawImage(baseCanvas, 0, 0);
    blurCtx.filter = 'none';
    blurCtx.globalAlpha = 0.65;
    blurCtx.drawImage(baseCanvas, 0, 0);
    return {
      url: blurCanvas.toDataURL('image/png'),
      width,
      height,
    };
  }

  return {
    url: baseCanvas.toDataURL('image/png'),
    width,
    height,
  };
};

export default function Globe({ data }: GlobeProps) {
  const cesiumContainer = useRef<HTMLDivElement>(null);
  const viewer = useRef<Cesium.Viewer | null>(null);
  const heatmapLayer = useRef<Cesium.ImageryLayer | null>(null);

  useEffect(() => {
    if (!cesiumContainer.current) return;

    // Set Cesium base URL for assets
    (window as unknown as { CESIUM_BASE_URL: string }).CESIUM_BASE_URL = '/cesium/';

    // Initialize Cesium viewer
    const initViewer = async () => {
      viewer.current = new Cesium.Viewer(cesiumContainer.current!, {
        homeButton: false,
        sceneModePicker: false,
        baseLayerPicker: false,
        navigationHelpButton: false,
        animation: false,
        timeline: false,
        fullscreenButton: false,
        vrButton: false,
        geocoder: false,
        selectionIndicator: false,
        infoBox: false,
        creditViewport: undefined, // Remove credit/watermark display
        // terrainProvider: await Cesium.createWorldTerrainAsync(), // Skip for now to avoid complexity
      });

      // Add realistic high-resolution Earth imagery with natural colors
      try {
        // Use OpenStreetMap satellite imagery (more reliable than ESRI)
        const osmSatelliteImagery = new Cesium.UrlTemplateImageryProvider({
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          maximumLevel: 18,
          credit: 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        });
        
        // Remove default imagery and add realistic satellite imagery
        viewer.current.imageryLayers.removeAll();
        viewer.current.imageryLayers.addImageryProvider(osmSatelliteImagery);
      } catch (error) {
        console.log('Falling back to standard OSM', error);
        // Fallback to standard OpenStreetMap if satellite fails
        try {
          const osmStandardImagery = new Cesium.OpenStreetMapImageryProvider({
            url: 'https://a.tile.openstreetmap.org/',
            maximumLevel: 18,
          });
          viewer.current.imageryLayers.removeAll();
          viewer.current.imageryLayers.addImageryProvider(osmStandardImagery);
        } catch (osmError) {
          console.log('Using default imagery', osmError);
        }
      }

      // Remove Cesium Ion watermark and credits
      if (viewer.current.cesiumWidget.creditContainer) {
        (viewer.current.cesiumWidget.creditContainer as HTMLElement).style.display = 'none';
      }
      
      // Also hide the credit display
      viewer.current.scene.primitives.removeAll();
      viewer.current.entities.removeAll();
      
      // Set dark theme but keep Earth imagery visible
      viewer.current.scene.backgroundColor = Cesium.Color.BLACK;
      // Remove the base color override to show natural Earth imagery
      // viewer.current.scene.globe.baseColor = Cesium.Color.BLACK;
      
      // Disable lighting for consistent brightness
      viewer.current.scene.globe.enableLighting = false;
      viewer.current.scene.globe.dynamicAtmosphereLighting = false;
      
      // Enable atmosphere for realistic look
      viewer.current.scene.globe.showGroundAtmosphere = true;
      if (viewer.current.scene.skyAtmosphere) {
        viewer.current.scene.skyAtmosphere.show = true;
      }

      // Set initial camera position (focused on world view)
      viewer.current.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000), // High altitude world view
        orientation: {
          heading: 0.0,
          pitch: -Cesium.Math.PI_OVER_TWO,
          roll: 0.0
        }
      });
    };

    initViewer();

    return () => {
      if (viewer.current) {
        if (heatmapLayer.current && viewer.current.imageryLayers.contains(heatmapLayer.current)) {
          viewer.current.imageryLayers.remove(heatmapLayer.current, true);
          heatmapLayer.current = null;
        }
        viewer.current.destroy();
        viewer.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!viewer.current || !data) return;

    const { lat_range, lon_range, prob_grid, statistics } = data;

    // Remove existing heatmap layer before rendering a new one
    if (heatmapLayer.current && viewer.current.imageryLayers.contains(heatmapLayer.current)) {
      viewer.current.imageryLayers.remove(heatmapLayer.current, true);
      heatmapLayer.current = null;
    }

    const heatmapTexture = createHeatmapTexture(lat_range, lon_range, prob_grid, statistics.max_probability);
    if (heatmapTexture) {
      const provider = new Cesium.SingleTileImageryProvider({
        url: heatmapTexture.url,
        rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
        tileWidth: heatmapTexture.width,
        tileHeight: heatmapTexture.height,
      });

      heatmapLayer.current = viewer.current.imageryLayers.addImageryProvider(provider);
      if (heatmapLayer.current) {
        heatmapLayer.current.alpha = 0.72;
        heatmapLayer.current.brightness = 1.08;
        heatmapLayer.current.contrast = 1.05;
      }
    }

    // Clear and rebuild entities (but we'll keep it empty for clean heatmap-only view)
    viewer.current.entities.removeAll();

    // Only show the smooth heatmap surface - no individual markers
    // This creates a clean, professional visualization

  }, [data]);

  return (
    <div 
      ref={cesiumContainer} 
      className="absolute inset-0 w-full h-full"
      style={{ background: 'black' }}
    />
  );
}