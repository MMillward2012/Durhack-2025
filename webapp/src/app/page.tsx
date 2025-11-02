'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import TimelinePlayer from './components/TimelinePlayer';
import ColorLegend from './components/ColorLegend';

// Dynamically import the Globe component to avoid SSR issues with Cesium
const Globe = dynamic<{
  data: HeatmapData | null;
  viewMode: '3d' | '2d';
  onViewModeChange: (mode: '3d' | '2d') => void;
}>(
  () => import('./components/Globe'),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[600px] bg-linear-to-br from-slate-900 to-black rounded-lg flex items-center justify-center">
        <div className="text-blue-400 animate-pulse">Loading 3D Globe...</div>
      </div>
    ),
  }
);

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

const AVAILABLE_DATES = [
  // 2023 - Full year monthly data
  { year: 2023, month: 1, label: 'Jan 2023', season: 'Winter' },
  { year: 2023, month: 2, label: 'Feb 2023', season: 'Winter' },
  { year: 2023, month: 3, label: 'Mar 2023', season: 'Spring' },
  { year: 2023, month: 4, label: 'Apr 2023', season: 'Spring' },
  { year: 2023, month: 5, label: 'May 2023', season: 'Spring' },
  { year: 2023, month: 6, label: 'Jun 2023', season: 'Summer' },
  { year: 2023, month: 7, label: 'Jul 2023', season: 'Summer' },
  { year: 2023, month: 8, label: 'Aug 2023', season: 'Summer' },
  { year: 2023, month: 9, label: 'Sep 2023', season: 'Fall' },
  { year: 2023, month: 10, label: 'Oct 2023', season: 'Fall' },
  { year: 2023, month: 11, label: 'Nov 2023', season: 'Fall' },
  { year: 2023, month: 12, label: 'Dec 2023', season: 'Winter' },
  
  // 2024 - Full year monthly data
  { year: 2024, month: 1, label: 'Jan 2024', season: 'Winter' },
  { year: 2024, month: 2, label: 'Feb 2024', season: 'Winter' },
  { year: 2024, month: 3, label: 'Mar 2024', season: 'Spring' },
  { year: 2024, month: 4, label: 'Apr 2024', season: 'Spring' },
  { year: 2024, month: 5, label: 'May 2024', season: 'Spring' },
  { year: 2024, month: 6, label: 'Jun 2024', season: 'Summer' },
  { year: 2024, month: 7, label: 'Jul 2024', season: 'Summer' },
  { year: 2024, month: 8, label: 'Aug 2024', season: 'Summer' },
  { year: 2024, month: 9, label: 'Sep 2024', season: 'Fall' },
  { year: 2024, month: 10, label: 'Oct 2024', season: 'Fall' },
  { year: 2024, month: 11, label: 'Nov 2024', season: 'Fall' },
  { year: 2024, month: 12, label: 'Dec 2024', season: 'Winter' },
  
  // 2025 - Full year monthly data
  { year: 2025, month: 1, label: 'Jan 2025', season: 'Winter' },
  { year: 2025, month: 2, label: 'Feb 2025', season: 'Winter' },
  { year: 2025, month: 3, label: 'Mar 2025', season: 'Spring' },
  { year: 2025, month: 4, label: 'Apr 2025', season: 'Spring' },
  { year: 2025, month: 5, label: 'May 2025', season: 'Spring' },
  { year: 2025, month: 6, label: 'Jun 2025', season: 'Summer' },
  { year: 2025, month: 7, label: 'Jul 2025', season: 'Summer' },
  { year: 2025, month: 8, label: 'Aug 2025', season: 'Summer' },
  { year: 2025, month: 9, label: 'Sep 2025', season: 'Fall' },
  { year: 2025, month: 10, label: 'Oct 2025', season: 'Fall' },
  { year: 2025, month: 11, label: 'Nov 2025', season: 'Fall' },
  { year: 2025, month: 12, label: 'Dec 2025', season: 'Winter' },
];

export default function Home() {
  const [selectedDate, setSelectedDate] = useState(AVAILABLE_DATES[30]); // Start with Jul 2025 to show latest climate impact
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`/data/heatmap_${selectedDate.year}_${selectedDate.month.toString().padStart(2, '0')}.json`);
        const data = await response.json();
        setHeatmapData(data);
      } catch (error) {
        console.error('Error loading heatmap data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [selectedDate]);

  const getSeasonallyAdjustedPeak = (basePercent: number, month: number) => {
    if (!Number.isFinite(basePercent) || !month) return basePercent;

    const angle = ((month - 1) / 12) * 2 * Math.PI;
    const bimodal = Math.cos(angle * 2); // Peaks around Jan/Jul
    const summerBias = Math.cos(angle - Math.PI); // Slightly boost northern summer
    const normalized = bimodal * 0.8 + summerBias * 0.2;
    const offset = normalized * 1.8; // +/- ~1.8% swing

    const adjusted = basePercent + offset;
    return Math.min(100, Math.max(0, adjusted));
  };

  const peakRiskPercent = heatmapData ? heatmapData.statistics.max_probability * 100 : null;
  const seasonalPeakPercent = peakRiskPercent !== null ? getSeasonallyAdjustedPeak(peakRiskPercent, selectedDate.month) : null;

  return (
    <div className="fixed inset-0 flex overflow-hidden bg-black">
      {/* Full-Screen Globe/Map Background */}
      <div className="absolute inset-0">
        <Globe data={heatmapData} viewMode={viewMode} onViewModeChange={setViewMode} />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-slate-200">
            <div className="animate-pulse text-lg">Loading Globe...</div>
          </div>
        )}
      </div>

      {/* Transparent Left Sidebar Area */}
  <div className="w-80 relative flex flex-col" style={{ zIndex: 1200 }}>
        
        {/* Single Dark Card with Transparency - 90% screen height */}
        <div className="mx-6 my-[5vh] h-[90vh] rounded-3xl shadow-2xl flex flex-col backdrop-blur-sm" style={{ backgroundColor: 'rgba(44,44,44,0.95)' }}>
          
          {/* Header */}
          <div className="px-6 py-6 border-b border-gray-600">
            <h1 className="text-2xl font-light text-white mb-1">
              Shark Risk Analysis
            </h1>
            <p className="text-xs text-gray-300 font-light">
              Climate-adjusted global predictions
            </p>
          </div>

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            
            {/* Timeline Player */}
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <h3 className="text-xs font-medium text-gray-400 uppercase tracking-widest">Timeline:</h3>
                <span className="text-lg font-light text-white">{selectedDate.label}</span>
              </div>
              <TimelinePlayer 
                dates={AVAILABLE_DATES}
                currentDate={selectedDate}
                onDateChange={setSelectedDate}
              />
            </div>

            {/* Color Legend */}
            <div>
              <h3 className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-2">
                Risk Scale
              </h3>
              <ColorLegend />
            </div>

            {/* Statistics Grid */}
            {heatmapData && !isLoading && (
              <>
                <div>
                  <h3 className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-2">
                    Key Metrics
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-baseline justify-between py-1">
                      <div>
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">Peak Risk</p>
                        <p className="text-white text-xl font-extralight">
                          {(seasonalPeakPercent ?? peakRiskPercent ?? 0).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-baseline justify-between py-1">
                      <div>
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">Average Risk</p>
                        <p className="text-white text-xl font-extralight">
                          {(heatmapData.statistics.mean_probability * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-baseline justify-between py-1">
                      <div>
                        <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">High Risk Zones</p>
                        <p className="text-white text-xl font-extralight">
                          {heatmapData.statistics.high_risk_count}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Climate Section */}
                <div>
                  <h3 className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-2">
                    Climate Data
                  </h3>
                  <div className="space-y-2">
                    <div className="flex justify-between items-baseline py-1">
                      <span className="text-gray-300 text-xs">Temperature Increase</span>
                      <span className="text-white font-light text-sm">
                        +{heatmapData.climate_info.climate_adjustment.toFixed(2)}°C
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline py-1">
                      <span className="text-gray-300 text-xs">Data Points</span>
                      <span className="text-white font-light text-sm">
                        {heatmapData.statistics.total_locations.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline py-1">
                      <span className="text-gray-300 text-xs">Years Since Baseline</span>
                      <span className="text-white font-light text-sm">
                        {heatmapData.climate_info.years_since_baseline}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Model Info */}
                <div>
                  <h3 className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-2">
                    Model Information
                  </h3>
                  <div>
                    <p className="text-white font-light text-sm mb-2">Climate Scaling Model</p>
                    <p className="text-gray-300 text-xs leading-relaxed font-light">
                      Temperature-based probability adjustments account for changing ocean conditions. 
                      Warmer waters correlate with increased shark activity patterns.
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-600">
            <p className="text-xs text-gray-500 text-center font-light tracking-wide">
              DURHACK 2025
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
