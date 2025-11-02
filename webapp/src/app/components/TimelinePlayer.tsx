'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

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
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1500); // milliseconds between changes
  
  const currentIndex = dates.findIndex(d => d.year === currentDate.year && d.month === currentDate.month);
  
  const nextDate = useCallback(() => {
    const nextIndex = (currentIndex + 1) % dates.length;
    onDateChange(dates[nextIndex]);
  }, [currentIndex, dates, onDateChange]);
  
  const prevDate = useCallback(() => {
    const prevIndex = currentIndex === 0 ? dates.length - 1 : currentIndex - 1;
    onDateChange(dates[prevIndex]);
  }, [currentIndex, dates, onDateChange]);
  
  const resetToStart = useCallback(() => {
    onDateChange(dates[0]);
    setIsPlaying(false);
  }, [dates, onDateChange]);
  
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying) {
      interval = setInterval(() => {
        const nextIndex = (currentIndex + 1) % dates.length;
        onDateChange(dates[nextIndex]);
        
        // Auto-pause at the end
        if (nextIndex === dates.length - 1) {
          setIsPlaying(false);
        }
      }, playbackSpeed);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, currentIndex, dates, onDateChange, playbackSpeed]);
  
  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold flex items-center space-x-2">
          <span>🎬</span>
          <span>Climate Timeline Player</span>
        </h3>
        
        <div className="flex items-center space-x-2">
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            className="bg-white/10 text-white rounded px-2 py-1 text-sm border border-white/20"
          >
            <option value={3000} className="bg-slate-800">Slow (3s)</option>
            <option value={1500} className="bg-slate-800">Normal (1.5s)</option>
            <option value={800} className="bg-slate-800">Fast (0.8s)</option>
          </select>
        </div>
      </div>
      
      {/* Timeline visualization */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-blue-300 text-sm">2020</span>
          <span className="text-blue-300 text-sm">2024</span>
        </div>
        <div className="relative">
          <div className="w-full h-2 bg-white/20 rounded-full">
            <div 
              className="h-2 bg-gradient-to-r from-blue-500 to-red-500 rounded-full transition-all duration-500 ease-in-out"
              style={{ width: `${((currentIndex + 1) / dates.length) * 100}%` }}
            />
          </div>
          <div 
            className="absolute top-0 w-4 h-4 bg-white rounded-full shadow-lg transform -translate-y-1 transition-all duration-500 ease-in-out"
            style={{ left: `calc(${(currentIndex / (dates.length - 1)) * 100}% - 8px)` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-blue-200">
          <span>Lower Climate Impact</span>
          <span>Higher Climate Impact</span>
        </div>
      </div>
      
      {/* Controls */}
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={resetToStart}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          title="Reset to start"
        >
          <RotateCcw className="w-4 h-4 text-white" />
        </button>
        
        <button
          onClick={prevDate}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          title="Previous date"
        >
          <SkipBack className="w-4 h-4 text-white" />
        </button>
        
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`p-3 rounded-lg transition-all transform hover:scale-105 ${
            isPlaying 
              ? 'bg-red-500 hover:bg-red-600' 
              : 'bg-green-500 hover:bg-green-600'
          }`}
          title={isPlaying ? 'Pause' : 'Play timeline'}
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-white ml-0.5" />
          )}
        </button>
        
        <button
          onClick={nextDate}
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          title="Next date"
        >
          <SkipForward className="w-4 h-4 text-white" />
        </button>
      </div>
      
      {/* Current status */}
      <div className="mt-4 text-center">
        <div className="text-white font-medium">{currentDate.label}</div>
        <div className="text-blue-200 text-sm">
          {currentIndex + 1} of {dates.length} • {currentDate.season}
          {isPlaying && <span className="animate-pulse ml-2">▶ Playing</span>}
        </div>
      </div>
    </div>
  );
}