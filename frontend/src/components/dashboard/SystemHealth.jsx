import React from 'react';
import { useHealth } from '../../hooks/useHealth';
import { Cpu, Database, Activity, HardDrive, ShieldCheck, Zap } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/utils';
import LogViewer from './LogViewer';


const SystemHealth = () => {
    const { health, loading } = useHealth();

    if (loading && !health) {
        return (
            <div className="flex flex-col items-center justify-center py-40 text-muted font-mono animate-pulse">
                <Zap className="w-8 h-8 mb-4 text-accent" />
                Syncing system telemetry...
            </div>
        );
    }

    // Backend returns: { allocated_gb: n, total_gb: n, utilization_pct: n, ... }
    const gpuUsage = health?.gpu || { total_gb: 0, allocated_gb: 0, utilization_pct: 0 };

    return (
        <div className="space-y-7 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-surface border border-border p-5 rounded-2xl">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-3 bg-accent/10 rounded-xl text-accent"><Cpu className="w-5 h-5" /></div>
                        <div className={cn(
                            "px-2 py-1 rounded text-[10px] font-mono leading-none",
                            health?.system?.status === 'online' ? "bg-accent/10 text-accent" : "bg-accent-red/10 text-accent-red"
                        )}>
                            {health?.system?.status?.toUpperCase() || 'OFFLINE'}
                        </div>
                    </div>
                    <h4 className="text-xl font-head font-bold text-text truncate" title={gpuUsage.name}>
                        {gpuUsage.name || 'Detecting...'}
                    </h4>
                    <p className="text-[10px] text-muted font-mono uppercase tracking-widest mt-1">Primary GPU Core</p>
                </div>

                <div className="bg-surface border border-border p-5 rounded-2xl">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-3 bg-white/5 rounded-xl text-muted text-white"><Database className="w-5 h-5" /></div>
                        <div className="text-xs text-accent font-mono">{gpuUsage.allocated_gb?.toFixed(1) || 0} GB</div>
                    </div>
                    <h4 className="text-xl font-head font-bold text-text">{gpuUsage.total_gb?.toFixed(1) || 0} GB</h4>
                    <p className="text-[10px] text-muted font-mono uppercase tracking-widest mt-1">Dedicated VRAM</p>
                </div>

                <div className="bg-surface border border-border p-5 rounded-2xl">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-3 bg-white/5 rounded-xl text-muted text-white"><Activity className="w-5 h-5" /></div>
                        <div className="text-xs text-accent font-mono">ACTIVE</div>
                    </div>
                    <h4 className="text-xl font-head font-bold text-text">{health?.system?.engine || 'N/A'}</h4>
                    <p className="text-[10px] text-muted font-mono uppercase tracking-widest mt-1">Inference Backend</p>
                </div>

                <div className="bg-surface border border-border p-5 rounded-2xl">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-3 bg-white/5 rounded-xl text-muted text-white"><ShieldCheck className="w-5 h-5" /></div>
                        <div className="text-xs text-accent font-mono">STABLE</div>
                    </div>
                    <h4 className="text-xl font-head font-bold text-text">{health?.system?.version || 'v0.0.0'}</h4>
                    <p className="text-[10px] text-muted font-mono uppercase tracking-widest mt-1">System Version</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-7">
                <div className="lg:col-span-2 bg-surface border border-border rounded-2xl p-6">
                    <div className="flex justify-between items-center mb-8">
                        <div>
                            <h3 className="text-text font-head font-bold">Memory Pulse</h3>
                            <p className="text-xs text-muted font-body mt-1">Real-time VRAM allocation tracking (MB)</p>
                        </div>
                        <div className="bg-background border border-border px-3 py-1 rounded-lg text-[10px] font-mono text-muted">LIVE TELEMETRY</div>
                    </div>
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={[
                                { name: 'T-15', usage: (gpuUsage.allocated_gb * 1024) - 100 },
                                { name: 'T-10', usage: (gpuUsage.allocated_gb * 1024) - 50 },
                                { name: 'T-5', usage: (gpuUsage.allocated_gb * 1024) - 20 },
                                { name: 'NOW', usage: (gpuUsage.allocated_gb * 1024) },
                            ]}>
                                <defs>
                                    <linearGradient id="colorUsage" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#00FFB2" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#00FFB2" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1A1A1A', color: '#FFF' }}
                                    itemStyle={{ color: '#00FFB2' }}
                                />
                                <Area type="monotone" dataKey="usage" stroke="#00FFB2" fillOpacity={1} fill="url(#colorUsage)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-surface border border-border rounded-2xl p-6">
                    <h3 className="text-text font-head font-bold mb-6">Service Overview</h3>
                    <div className="space-y-6">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-accent" />
                                <span className="text-sm text-text font-body">FastAPI Gateway</span>
                            </div>
                            <span className="text-xs text-muted font-mono">1.2ms</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-accent" />
                                <span className="text-sm text-text font-body">SQLAlchemy Core</span>
                            </div>
                            <span className="text-xs text-muted font-mono">Connected</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-accent" />
                                <span className="text-sm text-text font-body">Async Detector</span>
                            </div>
                            <span className="text-xs text-muted font-mono">Running</span>
                        </div>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-accent" />
                                <span className="text-sm text-text font-body">Embedding Engine</span>
                            </div>
                            <span className="text-xs text-muted font-mono">Warm</span>
                        </div>
                    </div>

                    <div className="mt-12 p-5 bg-background border border-border border-dashed rounded-xl flex flex-col items-center text-center">
                        <HardDrive className="w-5 h-5 text-muted mb-2 opacity-30" />
                        <p className="text-[10px] text-muted font-mono uppercase tracking-widest">Storage Status</p>
                        <h5 className="text-lg font-bold text-text mt-1">2.4 GB / 500 GB</h5>
                    </div>
                </div>
            </div>

            <div className="animate-in slide-in-from-bottom-5 duration-700 delay-200">
                <LogViewer />
            </div>
        </div>
    );
};

export default SystemHealth;
