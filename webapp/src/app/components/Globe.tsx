'use client';

import React, { useRef, useEffect, useState } from 'react';
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

  // Continuous normalization - no discrete bands!
  const normalizedProb = Math.min(probability / maxProbability, 1);
  
  // Only show colors for significant probabilities (reduced from 5% to 2.5%)
  if (normalizedProb < 0.025) return null; // Skip very low probability areas
  
  // Smooth heat map colors: light yellow -> orange -> red
  // Red channel: always high
  const r = 255;
  
  // Green channel: decreases from yellow to orange to red
  const g = Math.round(255 * Math.pow(1 - normalizedProb * 0.85, 1.2));
  
  // Blue channel: very low to maintain warm colors
  const b = Math.round(30 * Math.pow(1 - normalizedProb, 2));
  
  // Alpha: much more translucent (reduced from 0.3-0.8 to 0.15-0.5)
  const alpha = 0.15 + 0.35 * Math.pow(normalizedProb, 0.8);

  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
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

  // Bilinear interpolation function for smooth transitions
  const interpolateValue = (x: number, y: number): number => {
    // Convert canvas pixel to lat/lon
    // latRange goes from -90 to 90, lonRange goes from -180 to 180
    const lon = (x / width) * 360 - 180;
    const lat = 90 - (y / height) * 180;
    
    // Map lat/lon directly to grid indices
    // latRange is sorted from -90 to 90, so we can calculate directly
    const gridY = ((lat - latRange[0]) / (latRange[latRange.length - 1] - latRange[0])) * (latRange.length - 1);
    const gridX = ((lon - lonRange[0]) / (lonRange[lonRange.length - 1] - lonRange[0])) * (lonRange.length - 1);
    
    // Get integer grid positions
    const x1 = Math.floor(gridX);
    const y1 = Math.floor(gridY);
    const x2 = Math.min(x1 + 1, lonRange.length - 1);
    const y2 = Math.min(y1 + 1, latRange.length - 1);
    
    // Bounds check
    if (x1 < 0 || y1 < 0 || x2 >= lonRange.length || y2 >= latRange.length) return 0;
    
    // Get fractional parts
    const fx = gridX - x1;
    const fy = gridY - y1;
    
    // Get grid values (with bounds checking)
    const v11 = (y1 >= 0 && y1 < probGrid.length && x1 >= 0 && x1 < probGrid[y1].length) ? probGrid[y1][x1] || 0 : 0;
    const v12 = (y2 >= 0 && y2 < probGrid.length && x1 >= 0 && x1 < probGrid[y2].length) ? probGrid[y2][x1] || 0 : 0;
    const v21 = (y1 >= 0 && y1 < probGrid.length && x2 >= 0 && x2 < probGrid[y1].length) ? probGrid[y1][x2] || 0 : 0;
    const v22 = (y2 >= 0 && y2 < probGrid.length && x2 >= 0 && x2 < probGrid[y2].length) ? probGrid[y2][x2] || 0 : 0;
    
    // Apply cubic interpolation for smoother curves
    const smoothFx = fx * fx * (3 - 2 * fx); // Smoothstep function
    const smoothFy = fy * fy * (3 - 2 * fy);
    
    // Bilinear interpolation with smooth curves
    const top = v11 * (1 - smoothFx) + v21 * smoothFx;
    const bottom = v12 * (1 - smoothFx) + v22 * smoothFx;
    return top * (1 - smoothFy) + bottom * smoothFy;
  };

  // Multi-sample anti-aliasing for even smoother results
  const getSmoothedValue = (x: number, y: number): number => {
    // Use 4x supersampling for anti-aliasing
    const samples = [
      interpolateValue(x - 0.25, y - 0.25),
      interpolateValue(x + 0.25, y - 0.25),
      interpolateValue(x - 0.25, y + 0.25),
      interpolateValue(x + 0.25, y + 0.25)
    ];
    
    return samples.reduce((sum, val) => sum + val, 0) / samples.length;
  };

  // Render each pixel with interpolated values
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Use smoothed value with anti-aliasing for better quality
      const probability = getSmoothedValue(x, y);
      
      // Only render pixels where there's significant heatmap data
      if (probability > 0.001) {
        const colorString = getHeatmapColorString(probability, maxProbability);
        
        if (colorString) {
          // Parse the rgba color string
          const match = colorString.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/);
          if (match) {
            const pixelIndex = (y * width + x) * 4;
            data[pixelIndex] = parseInt(match[1]);     // Red
            data[pixelIndex + 1] = parseInt(match[2]); // Green
            data[pixelIndex + 2] = parseInt(match[3]); // Blue
            data[pixelIndex + 3] = Math.round(parseFloat(match[4]) * 255); // Alpha
          }
        }
      }
      // Pixels with probability <= 0.001 remain transparent (alpha=0)
      // This ensures Earth's natural colors show through where there's no risk
    }
  }
  
  // Put the image data on the canvas
  ctx.putImageData(imageData, 0, 0);

  // Use PNG format to preserve alpha channel transparency properly
  return {
    url: baseCanvas.toDataURL('image/png'),
    width,
    height
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

export default function Globe({ data }: GlobeProps) {
  const cesiumContainer = useRef<HTMLDivElement>(null);
  const viewer = useRef<Cesium.Viewer | null>(null);
  const heatmapLayer = useRef<Cesium.ImageryLayer | null>(null);
  const [clickInfo, setClickInfo] = useState<{ lat: number; lon: number; probability: number; percentage: string } | null>(null);

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

  // Separate useEffect for click handler that depends on data
  useEffect(() => {
    if (!viewer.current || !data) return;

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.current.scene.canvas);
    handler.setInputAction((movement: { position: Cesium.Cartesian2 }) => {
      if (!viewer.current || !data) return;

      const cartesian = viewer.current.camera.pickEllipsoid(movement.position, viewer.current.scene.globe.ellipsoid);
      if (cartesian) {
        const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
        const lat = Cesium.Math.toDegrees(cartographic.latitude);
        const lon = Cesium.Math.toDegrees(cartographic.longitude);

        // Get probability at this location
        const { lat_range, lon_range, prob_grid, statistics } = data;
        
        // Map lat/lon to grid indices
        const gridY = ((lat - lat_range[0]) / (lat_range[lat_range.length - 1] - lat_range[0])) * (lat_range.length - 1);
        const gridX = ((lon - lon_range[0]) / (lon_range[lon_range.length - 1] - lon_range[0])) * (lon_range.length - 1);
        
        const y = Math.round(gridY);
        const x = Math.round(gridX);
        
        if (y >= 0 && y < prob_grid.length && x >= 0 && x < prob_grid[y].length) {
          const probability = prob_grid[y][x] || 0;
          const percentage = ((probability / statistics.max_probability) * 100).toFixed(2);
          
          setClickInfo({
            lat: lat,
            lon: lon,
            probability: probability,
            percentage: percentage
          });
        }
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    return () => {
      handler.destroy();
    };
  }, [data]);

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
              // Don't apply layer-wide alpha - let our per-pixel alpha handle transparency
              heatmapLayer.current.alpha = 1.0; // Full alpha, our colors handle transparency
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
    <>
      <div 
        ref={cesiumContainer} 
        className="absolute inset-0 w-full h-full"
        style={{ background: 'black' }}
      />
      
      {/* Click info display */}
      {clickInfo && (
        <div className="absolute top-4 right-4 bg-black/80 text-white p-4 rounded-lg shadow-lg backdrop-blur-sm border border-white/20">
          <h3 className="font-bold text-lg mb-2">Shark Attack Risk</h3>
          <div className="space-y-1 text-sm">
            <p><span className="text-gray-400">Latitude:</span> {clickInfo.lat.toFixed(4)}°</p>
            <p><span className="text-gray-400">Longitude:</span> {clickInfo.lon.toFixed(4)}°</p>
            <p className="pt-2 border-t border-white/20">
              <span className="text-gray-400">Risk Level:</span>{' '}
              <span className="text-yellow-400 font-semibold">{clickInfo.percentage}%</span>
            </p>
            <p className="text-xs text-gray-500">
              Probability: {clickInfo.probability.toFixed(6)}
            </p>
          </div>
          <button 
            onClick={() => setClickInfo(null)}
            className="mt-3 w-full bg-white/10 hover:bg-white/20 text-white text-xs py-1 px-2 rounded transition-colors"
          >
            Close
          </button>
        </div>
      )}
    </>
  );
}