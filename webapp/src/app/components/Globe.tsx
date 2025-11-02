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

const getHeatmapColorString = (probability: number, _maxProbability: number): string | null => {
  if (!probability || probability <= 0.001) return null;

  // Use a reasonable threshold similar to the original
  const minThreshold = 0.02; // 2% minimum threshold
  if (probability < minThreshold) {
    return null; // Hide low probability areas to keep Earth visible
  }
  
  // Map probability 2%-100% to the YlOrRd colormap (like matplotlib)
  const normalizedProb = Math.min((probability - minThreshold) / (1 - minThreshold), 1);
  
  // Implement matplotlib's YlOrRd colormap colors
  let r, g, b, alpha;
  
  if (normalizedProb < 0.25) {
    // Light yellow to yellow (like YlOrRd start)
    const t = normalizedProb / 0.25;
    r = Math.round(255 * (1.0 - 0.1 * t)); // 255 to 230
    g = Math.round(255 * (1.0 - 0.2 * t)); // 255 to 204
    b = Math.round(255 * (0.8 - 0.6 * t)); // 204 to 51
    alpha = 0.5 + t * 0.2; // 0.5 to 0.7
  } else if (normalizedProb < 0.5) {
    // Yellow to orange
    const t = (normalizedProb - 0.25) / 0.25;
    r = Math.round(230 + 23 * t); // 230 to 253
    g = Math.round(204 - 60 * t); // 204 to 144
    b = Math.round(51 - 24 * t);  // 51 to 27
    alpha = 0.7 + t * 0.15; // 0.7 to 0.85
  } else if (normalizedProb < 0.75) {
    // Orange to red-orange
    const t = (normalizedProb - 0.5) / 0.25;
    r = Math.round(253 - 26 * t); // 253 to 227
    g = Math.round(144 - 80 * t); // 144 to 64
    b = Math.round(27 - 15 * t);  // 27 to 12
    alpha = 0.85 + t * 0.1; // 0.85 to 0.95
  } else {
    // Red-orange to deep red
    const t = (normalizedProb - 0.75) / 0.25;
    r = Math.round(227 - 47 * t); // 227 to 180
    g = Math.round(64 - 64 * t);  // 64 to 0
    b = Math.round(12 - 12 * t);  // 12 to 0
    alpha = 0.95 + t * 0.05; // 0.95 to 1.0
  }

  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
};

const createHeatmapTexture = (
  latRange: number[],
  lonRange: number[],
  probGrid: number[][],
  maxProbability: number
): { url: string; width: number; height: number } | null => {
  // Higher resolution for more precise heatmap details
  const width = 4096;
  const height = 2048;

  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = width;
  baseCanvas.height = height;

  const ctx = baseCanvas.getContext('2d');
  if (!ctx) return null;

  // Enable image smoothing for better performance with large datasets
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  
  // Clear canvas with full transparency
  ctx.clearRect(0, 0, width, height);

  const cellWidth = width / lonRange.length;
  const cellHeight = height / latRange.length;

  // Use ImageData for better performance with large datasets
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  
  // Initialize with transparent pixels
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 0;     // Red
    data[i + 1] = 0; // Green
    data[i + 2] = 0; // Blue
    data[i + 3] = 0; // Alpha (transparent)
  }

  // Process in batches to avoid blocking the main thread
  const processGrid = () => {
    for (let i = 0; i < latRange.length; i++) {
      for (let j = 0; j < lonRange.length; j++) {
        const probability = probGrid[i]?.[j];
        if (!probability || probability <= 0.001) continue;

        const colorString = getHeatmapColorString(probability, maxProbability);
        if (!colorString) continue;

        // Parse the RGBA color string to RGBA
        const rgba = parseRGBAToRGBA(colorString);
        if (!rgba) continue;

        const startX = Math.floor(j * cellWidth);
        const endX = Math.min(Math.ceil((j + 1) * cellWidth), width);
        const startY = Math.floor((latRange.length - 1 - i) * cellHeight);
        const endY = Math.min(Math.ceil((latRange.length - i) * cellHeight), height);

        // Only fill pixels where we have actual data - use precise cell boundaries
        const pixelStartX = Math.max(0, startX);
        const pixelEndX = Math.min(width, endX);
        const pixelStartY = Math.max(0, startY);
        const pixelEndY = Math.min(height, endY);
        
        for (let x = pixelStartX; x < pixelEndX; x++) {
          for (let y = pixelStartY; y < pixelEndY; y++) {
            const index = (y * width + x) * 4;
            data[index] = rgba.r;     // Red
            data[index + 1] = rgba.g; // Green
            data[index + 2] = rgba.b; // Blue
            data[index + 3] = rgba.a; // Alpha
          }
        }
      }
    }
  };

  processGrid();
  ctx.putImageData(imageData, 0, 0);

  // Apply light additional blur similar to the original scipy smoothing
  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const blurCtx = blurCanvas.getContext('2d');

  if (blurCtx) {
    // Light bilinear interpolation blur to match the original smooth appearance
    blurCtx.filter = 'blur(1.5px)';
    blurCtx.drawImage(baseCanvas, 0, 0);
    blurCtx.filter = 'none';
    blurCtx.globalAlpha = 0.85; // Blend with original for natural appearance
    blurCtx.drawImage(baseCanvas, 0, 0);
    return {
      url: blurCanvas.toDataURL('image/png', 0.9), // High quality PNG
      width,
      height,
    };
  }

  return {
    url: baseCanvas.toDataURL('image/jpeg', 0.8),
    width,
    height,
  };
};

// Gaussian smoothing function using proper 2D Gaussian filter (like scipy.ndimage.gaussian_filter)
const gaussianFilter2D = (probGrid: number[][], sigma: number = 1.5): number[][] => {
  const rows = probGrid.length;
  const cols = probGrid[0]?.length || 0;
  
  if (rows === 0 || cols === 0) return probGrid;
  
  // Calculate kernel size based on sigma (3 standard deviations)
  const kernelRadius = Math.ceil(3 * sigma);
  const kernelSize = 2 * kernelRadius + 1;
  
  // Create 2D Gaussian kernel
  const kernel: number[][] = [];
  let kernelSum = 0;
  
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] = [];
    for (let j = 0; j < kernelSize; j++) {
      const x = i - kernelRadius;
      const y = j - kernelRadius;
      const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel[i][j] = value;
      kernelSum += value;
    }
  }
  
  // Normalize kernel
  for (let i = 0; i < kernelSize; i++) {
    for (let j = 0; j < kernelSize; j++) {
      kernel[i][j] /= kernelSum;
    }
  }
  
  // Apply convolution with proper boundary handling
  const filtered: number[][] = [];
  for (let i = 0; i < rows; i++) {
    filtered[i] = [];
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      let weightSum = 0;
      
      for (let ki = 0; ki < kernelSize; ki++) {
        for (let kj = 0; kj < kernelSize; kj++) {
          const ni = i + ki - kernelRadius;
          const nj = j + kj - kernelRadius;
          
          // Handle boundaries by clamping (similar to scipy's 'nearest' mode)
          const clamped_i = Math.max(0, Math.min(rows - 1, ni));
          const clamped_j = Math.max(0, Math.min(cols - 1, nj));
          
          const value = probGrid[clamped_i][clamped_j];
          if (value !== null && value !== undefined) {
            sum += value * kernel[ki][kj];
            weightSum += kernel[ki][kj];
          }
        }
      }
      
      filtered[i][j] = weightSum > 0 ? sum / weightSum : probGrid[i][j] || 0;
    }
  }
  
  return filtered;
};

// Helper function to parse RGBA string to RGBA object
const parseRGBAToRGBA = (rgbaString: string): { r: number; g: number; b: number; a: number } | null => {
  const match = rgbaString.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/);
  if (!match) return null;

  return {
    r: parseInt(match[1]),
    g: parseInt(match[2]),
    b: parseInt(match[3]),
    a: Math.round(parseFloat(match[4]) * 255)
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
        const baseLayer = viewer.current.imageryLayers.addImageryProvider(osmSatelliteImagery);
        // Ensure base layer is visible and properly configured
        if (baseLayer) {
          baseLayer.alpha = 1.0;
          baseLayer.brightness = 1.0;
          baseLayer.contrast = 1.0;
        }
      } catch (error) {
        console.log('Falling back to standard OSM', error);
        // Fallback to standard OpenStreetMap if satellite fails
        try {
          const osmStandardImagery = new Cesium.OpenStreetMapImageryProvider({
            url: 'https://a.tile.openstreetmap.org/',
            maximumLevel: 18,
          });
          viewer.current.imageryLayers.removeAll();
          const fallbackLayer = viewer.current.imageryLayers.addImageryProvider(osmStandardImagery);
          if (fallbackLayer) {
            fallbackLayer.alpha = 1.0;
            fallbackLayer.brightness = 1.0;
            fallbackLayer.contrast = 1.0;
          }
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
      
      // Set background but keep Earth bright and visible
      viewer.current.scene.backgroundColor = Cesium.Color.BLACK;
      
      // Keep Earth imagery visible - disable lighting that might interfere
      viewer.current.scene.globe.enableLighting = false;
      viewer.current.scene.globe.dynamicAtmosphereLighting = false;
      
      // Don't override the base color - let the satellite imagery show through
      // viewer.current.scene.globe.baseColor = Cesium.Color.WHITE;
      
      // Enable atmosphere for realistic look
      viewer.current.scene.globe.showGroundAtmosphere = true;
      if (viewer.current.scene.skyAtmosphere) {
        viewer.current.scene.skyAtmosphere.show = true;
      }
      
      // Disable fog for cleaner view
      viewer.current.scene.fog.enabled = false;

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

    // Show loading state for large datasets
    console.log(`Processing heatmap with ${lat_range.length * lon_range.length} data points...`);
    
    // Use requestAnimationFrame to prevent blocking the UI
    const processHeatmap = () => {
      requestAnimationFrame(() => {
        try {
          // Apply minimal Gaussian smoothing with sigma=0.5 for sharp, precise points
          const smoothedGrid = gaussianFilter2D(prob_grid, 0.5);
          
          const heatmapTexture = createHeatmapTexture(lat_range, lon_range, smoothedGrid, statistics.max_probability);
          if (heatmapTexture && viewer.current) {
            const provider = new Cesium.SingleTileImageryProvider({
              url: heatmapTexture.url,
              rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
              tileWidth: heatmapTexture.width,
              tileHeight: heatmapTexture.height,
            });

            heatmapLayer.current = viewer.current.imageryLayers.addImageryProvider(provider);
            if (heatmapLayer.current) {
              heatmapLayer.current.alpha = 0.8; // Match original prediction.py alpha
              heatmapLayer.current.brightness = 1.0;
              heatmapLayer.current.contrast = 1.0;
              heatmapLayer.current.gamma = 1.0;
            }
          }
        } catch (error) {
          console.error('Error creating heatmap texture:', error);
        }
      });
    };

    processHeatmap();

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