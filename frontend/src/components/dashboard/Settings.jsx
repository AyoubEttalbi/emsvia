import React from 'react';
import { Save, Shield, Bell, Cpu, Database } from 'lucide-react';

const Settings = () => {
    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500 max-w-4xl">
            <div className="bg-surface border border-border rounded-2xl overflow-hidden">
                <div className="p-6 border-b border-border bg-white/2 flex items-center justify-between">
                    <div>
                        <h2 className="font-head text-lg font-bold text-text">System Configuration</h2>
                        <p className="text-xs text-muted font-mono uppercase mt-1">Global AI & Security Parameters</p>
                    </div>
                    <button className="bg-accent text-nav px-6 py-2 rounded-xl text-sm font-bold font-head hover:opacity-90 transition-all flex items-center gap-2">
                        <Save className="w-4 h-4" /> Save Changes
                    </button>
                </div>

                <div className="p-8 space-y-8">
                    {/* Section 1 */}
                    <div className="grid grid-cols-3 gap-8">
                        <div className="col-span-1">
                            <h3 className="text-sm font-semibold text-text flex items-center gap-2">
                                <Cpu className="w-4 h-4 text-accent" /> Engine
                            </h3>
                            <p className="text-xs text-muted mt-2 leading-relaxed font-body">Manage how the GPU-accelerated model processes frame buffers and face embeddings.</p>
                        </div>
                        <div className="col-span-2 space-y-4">
                            <div className="flex items-center justify-between p-4 bg-background border border-border rounded-xl">
                                <div>
                                    <p className="text-sm font-medium text-text">AI Vectorization Pipeline</p>
                                    <p className="text-[10px] text-muted font-mono">Sync student photos with the recognition engine</p>
                                </div>
                                <button
                                    onClick={async () => {
                                        try {
                                            const { rebuildEmbeddings } = await import('../../api/client');
                                            await rebuildEmbeddings();
                                            alert('Embedding generation started in background.');
                                        } catch (err) {
                                            alert('Failed to start pipeline');
                                        }
                                    }}
                                    className="bg-accent/10 text-accent border border-accent/20 px-4 py-2 rounded-lg text-[10px] font-bold font-mono hover:bg-accent hover:text-nav transition-all"
                                >
                                    START REBUILD
                                </button>
                            </div>
                            <div className="flex items-center justify-between p-4 bg-background border border-border rounded-xl">
                                <div>
                                    <p className="text-sm font-medium text-text">Detection Confidence</p>
                                    <p className="text-[10px] text-muted font-mono">Current: 0.85 (High Precision)</p>
                                </div>
                                <input type="range" className="accent-accent" />
                            </div>
                            <div className="flex items-center justify-between p-4 bg-background border border-border rounded-xl">
                                <div>
                                    <p className="text-sm font-medium text-text">Mixed Precision (FP16)</p>
                                    <p className="text-[10px] text-muted font-mono">Improves performance on NVIDIA GPUs</p>
                                </div>
                                <div className="w-10 h-5 bg-accent/20 rounded-full relative cursor-pointer">
                                    <div className="absolute right-1 top-1 w-3 h-3 bg-accent rounded-full" />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="h-[1px] bg-border" />

                    {/* Section 2 */}
                    <div className="grid grid-cols-3 gap-8">
                        <div className="col-span-1">
                            <h3 className="text-sm font-semibold text-text flex items-center gap-2">
                                <Shield className="w-4 h-4 text-accent-red" /> Security
                            </h3>
                            <p className="text-xs text-muted mt-2 leading-relaxed font-body">Data retention and privacy protocols for unreviewed captures.</p>
                        </div>
                        <div className="col-span-2 space-y-4">
                            <div className="flex items-center justify-between p-4 bg-background border border-border rounded-xl">
                                <div>
                                    <p className="text-sm font-medium text-text">Auto-Delete Unknowns</p>
                                    <p className="text-[10px] text-muted font-mono">Delete unreviewed faces after 7 days</p>
                                </div>
                                <div className="w-10 h-5 bg-white/5 rounded-full relative cursor-pointer border border-border">
                                    <div className="absolute left-1 top-1 w-3 h-3 bg-muted rounded-full" />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Settings;
