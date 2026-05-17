import React from 'react';
import MetricCard from './MetricCard';
import ActivityLog from './ActivityLog';
import UnknownFaceQueue from './UnknownFaceQueue';
import SystemHealth from './SystemHealth';
import { useStats } from '../../hooks/useStats';
import { useActivity } from '../../hooks/useActivity';
import { useUnknowns } from '../../hooks/useUnknowns';
import { 
  BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell, 
  PieChart, Pie, AreaChart, Area 
} from 'recharts';
import { 
  Users, CheckCircle2, AlertTriangle, Cpu, 
  TrendingUp, TrendingDown, Activity, CalendarDays,
  ChevronRight, Zap, Shield, Clock
} from 'lucide-react';

const Overview = () => {
  const { stats, loading: statsLoading } = useStats();
  const { activities, loading: activityLoading } = useActivity();
  const { unknowns, loading: unknownsLoading } = useUnknowns();

  const trendData = [
    { name: 'Mon', value: 198, full: 'Monday' },
    { name: 'Tue', value: 212, full: 'Tuesday' },
    { name: 'Wed', value: 176, full: 'Wednesday' },
    { name: 'Thu', value: 224, full: 'Thursday' },
    { name: 'Fri', value: 184, full: 'Friday' },
    { name: 'Sat', value: 0, full: 'Saturday' },
    { name: 'Sun', value: 0, full: 'Sunday' },
  ];

  const statusData = [
    { name: 'Present', value: stats?.attendance_today || 0, color: '#00e5b4' },
    { name: 'Late', value: 23, color: '#f5a623' },
    { name: 'Absent', value: 40, color: '#ff6b6b' },
  ];

  const metrics = [
    {
      label: "Total Enrolled",
      value: stats?.total_students,
      subtext: "Active students",
      icon: Users,
      color: "emerald",
      trend: "up",
      trendValue: "+12%"
    },
    {
      label: "Present Today",
      value: stats?.attendance_today,
      subtext: "Verified entries",
      icon: CheckCircle2,
      color: "cyan",
      trend: "up",
      trendValue: "+5%"
    },
    {
      label: "Pending Reviews",
      value: stats?.unknown_faces_pending,
      subtext: "Needs attention",
      icon: AlertTriangle,
      color: "amber",
      trend: stats?.unknown_faces_pending > 0 ? 'down' : 'up',
      critical: stats?.unknown_faces_pending > 5
    },
    {
      label: "GPU Engine",
      value: stats?.gpu_active ? "READY" : "CPU",
      subtext: stats?.gpu_active ? "NVIDIA Active" : "Fallback mode",
      icon: stats?.gpu_active ? Zap : Cpu,
      color: stats?.gpu_active ? "teal" : "slate",
      badge: stats?.gpu_active ? "CUDA 12.1" : null
    }
  ];

  const colorMap = {
    emerald: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      text: 'text-emerald-400',
      icon: 'text-emerald-400',
      glow: 'shadow-emerald-500/20'
    },
    cyan: {
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/20',
      text: 'text-cyan-400',
      icon: 'text-cyan-400',
      glow: 'shadow-cyan-500/20'
    },
    amber: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
      text: 'text-amber-400',
      icon: 'text-amber-400',
      glow: 'shadow-amber-500/20'
    },
    teal: {
      bg: 'bg-teal-500/10',
      border: 'border-teal-500/20',
      text: 'text-teal-400',
      icon: 'text-teal-400',
      glow: 'shadow-teal-500/20'
    },
    slate: {
      bg: 'bg-slate-500/10',
      border: 'border-slate-500/20',
      text: 'text-slate-400',
      icon: 'text-slate-400',
      glow: 'shadow-slate-500/20'
    }
  };

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto pb-10 p-6">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Dashboard Overview</h1>
          <p className="text-sm text-slate-400 mt-1 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-teal-400" />
            Real-time biometric attendance monitoring
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-surface border border-border rounded-lg">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-mono text-emerald-400">SYSTEM ONLINE</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => {
          const colors = colorMap[metric.color];
          const Icon = metric.icon;
          
          return (
            <div 
              key={metric.label}
              className={`group relative bg-surface border ${colors.border} rounded-xl p-5 
                hover:border-opacity-50 transition-all duration-300 hover:shadow-lg ${colors.glow}
                ${metric.critical ? 'ring-1 ring-amber-500/30' : ''}`}
            >
              {/* Background gradient on hover */}
              <div className={`absolute inset-0 ${colors.bg} opacity-0 group-hover:opacity-100 
                rounded-xl transition-opacity duration-300`} 
              />
              
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-2.5 rounded-lg ${colors.bg} ${colors.border} border`}>
                    <Icon className={`w-5 h-5 ${colors.icon}`} strokeWidth={2} />
                  </div>
                  
                  {metric.trend && (
                    <div className={`flex items-center gap-1 text-xs font-mono 
                      ${metric.trend === 'up' ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {metric.trend === 'up' ? (
                        <TrendingUp className="w-3 h-3" />
                      ) : (
                        <TrendingDown className="w-3 h-3" />
                      )}
                      {metric.trendValue}
                    </div>
                  )}
                  
                  {metric.badge && (
                    <span className="px-2 py-0.5 bg-teal-500/10 border border-teal-500/20 
                      rounded text-[10px] font-mono text-teal-400">
                      {metric.badge}
                    </span>
                  )}
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">
                    {metric.label}
                  </p>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-3xl font-bold font-mono tracking-tight text-white`}>
                      {statsLoading ? (
                        <span className="animate-pulse text-slate-600">---</span>
                      ) : (
                        metric.value ?? 0
                      )}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 flex items-center gap-1.5 mt-2">
                    {metric.critical && (
                      <Shield className="w-3 h-3 text-amber-400" />
                    )}
                    {metric.subtext}
                  </p>
                </div>
              </div>

              {/* Decorative corner accent */}
              <div className={`absolute top-0 right-0 w-16 h-16 opacity-10 ${colors.text}`}>
                <Icon className="w-full h-full -translate-y-1/2 translate-x-1/2" strokeWidth={0.5} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-[2fr,1fr] gap-4">
        
        {/* Attendance Trend Chart */}
        <div className="bg-surface border border-border rounded-xl p-6 flex flex-col h-[400px] relative overflow-hidden">
          {/* Subtle grid background */}
          <div className="absolute inset-0 opacity-[0.02]" 
            style={{ 
              backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
              backgroundSize: '20px 20px'
            }} 
          />

          <div className="relative z-10 flex justify-between items-start mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <CalendarDays className="w-4 h-4 text-teal-400" />
                <h3 className="font-semibold text-white text-sm">Attendance Trend</h3>
              </div>
              <p className="text-xs text-slate-500 font-mono">Last 7 days · check-ins</p>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 bg-white/5 border border-border rounded-md text-[10px] text-slate-400 font-mono">
                weekly
              </span>
              <button className="p-1.5 hover:bg-white/5 rounded-md transition-colors">
                <ChevronRight className="w-4 h-4 text-slate-500" />
              </button>
            </div>
          </div>

          {/* Filter Tabs */}
          <div className="relative z-10 flex gap-2 mb-6">
            {['All', 'Present', 'Late', 'Absent'].map((tab, i) => (
              <button 
                key={tab} 
                className={`px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-200
                  ${i === 0 
                    ? 'bg-teal-500/10 border border-teal-500/20 text-teal-400 shadow-sm' 
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200 border border-transparent'
                  }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="relative z-10 flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trendData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00e5b4" stopOpacity={0.8}/>
                    <stop offset="100%" stopColor="#00e5b4" stopOpacity={0.2}/>
                  </linearGradient>
                  <linearGradient id="barGradientDim" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00e5b4" stopOpacity={0.3}/>
                    <stop offset="100%" stopColor="#00e5b4" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="name" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }} 
                  dy={10}
                />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.03)' }} 
                  contentStyle={{ 
                    backgroundColor: '#0f1117', 
                    border: '1px solid rgba(255,255,255,0.1)', 
                    borderRadius: '12px',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
                    padding: '12px'
                  }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '12px', fontFamily: 'JetBrains Mono, monospace' }}
                  labelStyle={{ color: '#64748b', fontSize: '11px', marginBottom: '4px' }}
                />
                <Bar 
                  dataKey="value" 
                  radius={[6, 6, 0, 0]} 
                  barSize={32}
                  animationDuration={1500}
                >
                  {trendData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={index === 4 ? 'url(#barGradient)' : index >= 5 ? 'rgba(255,255,255,0.03)' : 'url(#barGradientDim)'}
                      stroke={index === 4 ? '#00e5b4' : 'none'}
                      strokeWidth={index === 4 ? 1 : 0}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart footer stats */}
          <div className="relative z-10 flex items-center justify-between mt-4 pt-4 border-t border-border">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-teal-400" />
                <span className="text-[10px] text-slate-500 font-mono">AVG: 201/day</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-amber-400" />
                <span className="text-[10px] text-slate-500 font-mono">PEAK: Thu</span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-[10px] text-slate-600 font-mono">
              <Clock className="w-3 h-3" />
              Updated 2m ago
            </div>
          </div>
        </div>

        {/* Status Split Chart */}
        <div className="bg-surface border border-border rounded-xl p-6 flex flex-col h-[400px] relative overflow-hidden">
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-1">
              <Shield className="w-4 h-4 text-teal-400" />
              <h3 className="font-semibold text-white text-sm">Status Split</h3>
            </div>
            <p className="text-xs text-slate-500 font-mono">Today's distribution</p>
          </div>

          <div className="flex-1 flex flex-col items-center justify-center relative">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <defs>
                  {statusData.map((entry, index) => (
                    <linearGradient key={`grad-${index}`} id={`pieGrad-${index}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={entry.color} stopOpacity={1}/>
                      <stop offset="100%" stopColor={entry.color} stopOpacity={0.6}/>
                    </linearGradient>
                  ))}
                </defs>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={52}
                  outerRadius={68}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                  animationDuration={1500}
                  animationBegin={200}
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`url(#pieGrad-${index})`} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            
            {/* Center content */}
            <div className="absolute top-[40%] left-[50%] -translate-x-1/2 text-center pointer-events-none">
              <div className="text-2xl font-bold text-white font-mono tracking-tight">
                {stats?.attendance_today || 0}
              </div>
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-widest mt-0.5">
                present
              </div>
            </div>

            {/* Legend */}
            <div className="w-full mt-8 space-y-3">
              {statusData.map((item, idx) => (
                <div key={item.name} className="group flex items-center p-2 rounded-lg hover:bg-white/5 transition-colors cursor-pointer">
                  <div className="relative">
                    <div 
                      className="w-3 h-3 rounded-[4px] mr-3 shadow-lg" 
                      style={{ backgroundColor: item.color, boxShadow: `0 0 10px ${item.color}40` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400 font-medium">{item.name}</span>
                  <span className="ml-auto font-mono text-sm text-white group-hover:text-teal-400 transition-colors">
                    {item.value}
                  </span>
                  <span className="ml-2 text-[10px] text-slate-600 font-mono">
                    {Math.round((item.value / (statusData.reduce((a, b) => a + b.value, 0) || 1)) * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Panels Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ActivityLog data={activities} loading={activityLoading} />
        <UnknownFaceQueue data={unknowns} loading={unknownsLoading} count={stats?.unknown_faces_pending} />
      </div>

      {/* System Health Panel */}
      <SystemHealth stats={stats} />
    </div>
  );
};

export default Overview;