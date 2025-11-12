'use client';

import React from 'react';

interface TimelinePlayerProps {
  dates: Array<{
    year: number;
    month: number;
    label: string;
    season: string;
  }>;
  currentDate: {
    year: number;
    month: number;
    label: string;
    season: string;
  };
  onDateChange: (date: {
    year: number;
    month: number;
    label: string;
    season: string;
  }) => void;
}

export default function TimelinePlayer({ dates, currentDate, onDateChange }: TimelinePlayerProps) {
  const currentIndex = dates.findIndex(d => d.year === currentDate.year && d.month === currentDate.month);
  
  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = parseInt(event.target.value);
    onDateChange(dates[newIndex]);
  };
  
  return (
    <div>
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
      `}</style>
      {/* Timeline visualization */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-gray-400 text-xs">2023</span>
          <span className="text-gray-400 text-xs">2025</span>
        </div>
        <div className="relative">
          <input
            type="range"
            min="0"
            max={dates.length - 1}
            value={currentIndex}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-600 rounded-full appearance-none cursor-pointer slider"
            style={{
              background: `linear-gradient(to right, #3b82f6 0%, #ef4444 100%)`
            }}
          />
        </div>
      </div>
    </div>
  );
}