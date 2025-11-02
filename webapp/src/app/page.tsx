'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import TimelinePlayer from './components/TimelinePlayer';

// Dynamically import the Globe component to avoid SSR issues with Cesium
const Globe = dynamic(
  () => import('./components/Globe'),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[600px] bg-gradient-to-br from-slate-900 to-black rounded-lg flex items-center justify-center">
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
  { year: 2020, month: 1, label: 'Jan 2020', season: 'Winter' },
  { year: 2020, month: 6, label: 'Jun 2020', season: 'Summer' },
  { year: 2020, month: 12, label: 'Dec 2020', season: 'Winter' },
  { year: 2021, month: 6, label: 'Jun 2021', season: 'Summer' },
  { year: 2021, month: 12, label: 'Dec 2021', season: 'Winter' },
  { year: 2022, month: 6, label: 'Jun 2022', season: 'Summer' },
  { year: 2022, month: 7, label: 'Jul 2022', season: 'Summer' },
  { year: 2022, month: 12, label: 'Dec 2022', season: 'Winter' },
  { year: 2023, month: 6, label: 'Jun 2023', season: 'Summer' },
  { year: 2023, month: 12, label: 'Dec 2023', season: 'Winter' },
  { year: 2024, month: 6, label: 'Jun 2024', season: 'Summer' },
  { year: 2024, month: 12, label: 'Dec 2024', season: 'Winter' },
];

export default function Home() {
  const [selectedDate, setSelectedDate] = useState(AVAILABLE_DATES[10]); // Start with 2024 June to show climate impact
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

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

  return (
    <div className="fixed inset-0 flex overflow-hidden bg-black">
      {/* Transparent Left Sidebar Area */}
      <div className="w-96 relative z-10 flex flex-col">
        
        {/* Single Dark Card - 90% screen height */}
        <div className="mx-6 my-[5vh] h-[90vh] bg-gray-950/95 backdrop-blur-xl rounded-3xl border border-gray-800/40 shadow-2xl flex flex-col">
          
          {/* Header */}
          <div className="px-8 py-8 border-b border-gray-800/30">
            <h1 className="text-3xl font-light text-white mb-2">
              Shark Risk Analysis
            </h1>
            <p className="text-sm text-gray-400 font-light">
              Climate-adjusted global predictions
            </p>
          </div>

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto px-8 py-8 space-y-8">
            
            {/* Timeline Player */}
            <div>
              <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-4">
                Timeline
              </h3>
              <TimelinePlayer 
                dates={AVAILABLE_DATES}
                currentDate={selectedDate}
                onDateChange={setSelectedDate}
              />
            </div>

            {/* Statistics Grid */}
            {heatmapData && !isLoading && (
              <>
                <div>
                  <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-4">
                    Key Metrics
                  </h3>
                  <div className="space-y-4">
                    <div className="bg-gray-900/60 rounded-2xl p-6 border border-gray-800/50">
                      <div className="flex items-baseline justify-between">
                        <div>
                          <p className="text-gray-500 text-xs font-medium uppercase tracking-wide">Peak Risk</p>
                          <p className="text-white text-2xl font-extralight mt-1">
                            {(heatmapData.statistics.max_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-gray-700 text-right">
                          <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-900/60 rounded-2xl p-6 border border-gray-800/50">
                      <div className="flex items-baseline justify-between">
                        <div>
                          <p className="text-gray-500 text-xs font-medium uppercase tracking-wide">Average Risk</p>
                          <p className="text-white text-2xl font-extralight mt-1">
                            {(heatmapData.statistics.mean_probability * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div className="text-gray-700 text-right">
                          <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-900/60 rounded-2xl p-6 border border-gray-800/50">
                      <div className="flex items-baseline justify-between">
                        <div>
                          <p className="text-gray-500 text-xs font-medium uppercase tracking-wide">High Risk Zones</p>
                          <p className="text-white text-2xl font-extralight mt-1">
                            {heatmapData.statistics.high_risk_count}
                          </p>
                        </div>
                        <div className="text-gray-700 text-right">
                          <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Climate Section */}
                <div>
                  <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-4">
                    Climate Data
                  </h3>
                  <div className="bg-gray-900/60 rounded-2xl p-6 border border-gray-800/50 space-y-5">
                    <div className="flex justify-between items-baseline">
                      <span className="text-gray-400 text-sm">Temperature Increase</span>
                      <span className="text-white font-light">
                        +{heatmapData.climate_info.climate_adjustment.toFixed(2)}°C
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-gray-400 text-sm">Data Points</span>
                      <span className="text-white font-light">
                        {heatmapData.statistics.total_locations.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-gray-400 text-sm">Years Since Baseline</span>
                      <span className="text-white font-light">
                        {heatmapData.climate_info.years_since_baseline}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Model Info */}
                <div>
                  <div className="bg-gray-900/40 rounded-2xl p-6 border border-gray-800/30">
                    <p className="text-white font-light text-sm mb-3">Climate Scaling Model</p>
                    <p className="text-gray-400 text-xs leading-relaxed font-light">
                      Temperature-based probability adjustments account for changing ocean conditions. 
                      Warmer waters correlate with increased shark activity patterns.
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="px-8 py-6 border-t border-gray-800/30">
            <p className="text-xs text-gray-600 text-center font-light tracking-wide">
              DURHACK 2025
            </p>
          </div>
        </div>
      </div>

      {/* Full-Screen Globe - Shifted Left */}
      <div className="flex-1 relative -ml-24">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black">
            <div className="text-slate-400 animate-pulse text-lg">Loading 3D Globe...</div>
          </div>
        ) : (
          <Globe data={heatmapData} />
        )}
      </div>
    </div>
  );
}
