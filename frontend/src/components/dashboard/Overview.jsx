import React from 'react';
import MetricCard from './MetricCard';
import ActivityLog from './ActivityLog';
import UnknownFaceQueue from './UnknownFaceQueue';
import SystemHealth from './SystemHealth';
import { useStats } from '../../hooks/useStats';
import { useActivity } from '../../hooks/useActivity';
import { useUnknowns } from '../../hooks/useUnknowns';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

const Overview = () => {
    const { stats, loading: statsLoading } = useStats();
    const { activities, loading: activityLoading } = useActivity();
    const { unknowns, loading: unknownsLoading } = useUnknowns();

    const trendData = [
        { name: 'Mon', value: 198 },
        { name: 'Tue', value: 212 },
        { name: 'Wed', value: 176 },
        { name: 'Thu', value: 224 },
        { name: 'Fri', value: 184 },
        { name: 'Sat', value: 0 },
        { name: 'Sun', value: 0 },
    ];

    const statusData = [
        { name: 'Present', value: stats?.attendance_today || 0, color: '#00e5b4' },
        { name: 'Late', value: 23, color: '#f5a623' },
        { name: 'Absent', value: 40, color: '#ff6b6b' },
    ];

    return (
        <div className="space-y-4 max-w-[1400px] mx-auto pb-10">
            {/* Metrics Row */}
            <div className="grid grid-cols-4 gap-[14px]">
                <MetricCard label="Total Enrolled" value={statsLoading ? "..." : stats?.total_students} delta="Active students" icon="📋" color="green" />
                <MetricCard label="Present Today" value={statsLoading ? "..." : stats?.attendance_today} delta="Verified entries" icon="✅" color="blue" />
                <MetricCard label="Pending Reviews" value={statsLoading ? "..." : stats?.unknown_faces_pending} delta="Needs attention" icon="⚠️" color="red" trend={stats?.unknown_faces_pending > 0 ? 'down' : 'up'} />
                <MetricCard label="GPU Engine" value={statsLoading ? "..." : (stats?.gpu_active ? "READY" : "CPU")} delta={stats?.gpu_active ? "NVIDIA Active" : "Fallback mode"} icon="🎯" color="amber" />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-[2fr,1fr] gap-[14px]">
                <div className="bg-surface border border-border rounded-xl p-5 flex flex-col h-[360px]">
                    <div className="flex justify-between items-center mb-6">
                        <div>
                            <h3 className="font-head text-[14px] font-semibold text-text">Attendance Trend</h3>
                            <p className="text-[11px] text-muted mt-[1px] font-mono">Last 7 days · check-ins</p>
                        </div>
                        <span className="bg-white/5 border border-border rounded-[6px] px-2 py-[2px] text-[10px] text-muted-2 font-mono">weekly</span>
                    </div>

                    <div className="flex gap-[6px] mb-[18px]">
                        {['All', 'Present', 'Late'].map((tab, i) => (
                            <div key={tab} className={`px-3 py-1 rounded-[6px] text-[11px] font-mono cursor-pointer transition-all ${i === 0 ? 'bg-accent/10 border border-accent/20 text-accent' : 'text-muted-2 hover:bg-white/5 hover:text-text'}`}>
                                {tab}
                            </div>
                        ))}
                    </div>

                    <div className="flex-1">
                        <ResponsiveContainer width="100%" height="100%">
                            < BarChart data={trendData}>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'DM Mono' }} dy={10} />
                                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.02)' }} contentStyle={{ backgroundColor: '#1a1e28', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '8px' }} />
                                < Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={24}>
                                    {trendData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={index === 4 ? '#00e5b4' : index >= 5 ? 'rgba(255,255,255,0.04)' : 'rgba(0,229,180,0.4)'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-surface border border-border rounded-xl p-5 flex flex-col h-[360px]">
                    <div className="mb-6">
                        <h3 className="font-head text-[14px] font-semibold text-text">Status Split</h3>
                        <p className="text-[11px] text-muted mt-[1px] font-mono">Today's distribution</p>
                    </div>

                    <div className="flex-1 flex flex-col items-center justify-center relative">
                        <ResponsiveContainer width="100%" height={160}>
                            <PieChart>
                                <Pie
                                    data={statusData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={48}
                                    outerRadius={60}
                                    paddingAngle={4}
                                    dataKey="value"
                                    stroke="none"
                                >
                                    {statusData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="absolute top-[40%] left-[50%] -translate-x-1/2 text-center pointer-events-none">
                            <div className="font-head text-base font-bold text-text leading-tight">{stats?.attendance_today || 0}</div>
                            <div className="text-[9px] text-muted font-mono uppercase tracking-tighter">present</div>
                        </div>

                        <div className="w-full mt-6 space-y-[6px]">
                            {statusData.map(item => (
                                <div key={item.name} className="flex items-center text-[10px] text-muted-2">
                                    <span className="w-2 h-2 rounded-[2px] mr-2" style={{ backgroundColor: item.color }} />
                                    {item.name}
                                    <span className="ml-auto font-mono text-text">{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Panels Row */}
            <div className="grid grid-cols-2 gap-[14px]">
                <ActivityLog data={activities} loading={activityLoading} />
                <UnknownFaceQueue data={unknowns} loading={unknownsLoading} count={stats?.unknown_faces_pending} />
            </div>

            {/* System Health Panel */}
            <SystemHealth stats={stats} />
        </div>
    );
};

export default Overview;
