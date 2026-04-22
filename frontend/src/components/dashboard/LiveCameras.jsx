import React, { useState, useEffect } from 'react';
import { Play, Square, Maximize2, Shield, Activity, Wifi, Settings, RefreshCw } from 'lucide-react';
import axios from 'axios';
import { cn } from '../../lib/utils';

const LiveCameras = () => {
    const [availableSources, setAvailableSources] = useState([]);
    const [detecting, setDetecting] = useState(false);
    const [cameraStates, setCameraStates] = useState({
        'cam-01': { isLive: false, index: 0, name: 'Entrance View' },
        'cam-02': { isLive: false, index: 1, name: 'Security Desk' },
    });

    const detectSources = async () => {
        setDetecting(true);
        try {
            const resp = await axios.get('http://127.0.0.1:8000/api/cameras/detect');
            setAvailableSources(resp.data);
        } catch (err) {
            console.error(err);
        } finally {
            setDetecting(false);
        }
    };

    useEffect(() => {
        detectSources();
    }, []);

    const toggleCamera = (id) => {
        setCameraStates(prev => ({
            ...prev,
            [id]: { ...prev[id], isLive: !prev[id].isLive }
        }));
    };

    const updateIndex = (id, newIdx) => {
        setCameraStates(prev => ({
            ...prev,
            [id]: { ...prev[id], index: parseInt(newIdx) }
        }));
    };

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="flex justify-between items-center bg-surface border border-border p-4 rounded-xl">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1 bg-accent/10 border border-accent/20 rounded-lg text-accent text-xs font-mono">
                        <Wifi className="w-3 h-3" />
                        <span>Hardware Sync</span>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1 bg-white/5 border border-border rounded-lg text-muted text-xs font-mono">
                        <Activity className="w-3 h-3" />
                        <span>Found {availableSources.length} video streams</span>
                    </div>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={detectSources}
                        disabled={detecting}
                        className="bg-white/5 text-text border border-border px-4 py-2 rounded-lg text-xs font-bold font-head hover:bg-white/10 transition-all flex items-center gap-2 disabled:opacity-50"
                    >
                        <RefreshCw className={cn("w-3 h-3", detecting && "animate-spin")} />
                        {detecting ? 'Scanning...' : 'Rescan Hardware'}
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {Object.entries(cameraStates).map(([id, cam]) => (
                    <div key={id} className="bg-surface border border-border rounded-2xl overflow-hidden group">
                        <div className="aspect-video bg-black relative overflow-hidden flex items-center justify-center">
                            {cam.isLive ? (
                                <img
                                    src={`http://127.0.0.1:8000/api/cameras/stream?cam_id=${cam.index}`}
                                    className="w-full h-full object-cover"
                                    alt={cam.name}
                                    key={`${id}-${cam.isLive}-${cam.index}`}
                                    onError={(e) => {
                                        alert(`Source Index ${cam.index} failed. It might be in use by another application.`);
                                        toggleCamera(id);
                                    }}
                                />
                            ) : (
                                <div className="flex flex-col items-center justify-center text-muted space-y-3">
                                    <Shield className="w-10 h-10 opacity-20" />
                                    <p className="text-[10px] font-mono uppercase tracking-widest opacity-40">Feed Standby</p>
                                    <button
                                        onClick={() => toggleCamera(id)}
                                        className="flex items-center gap-2 bg-accent text-nav px-6 py-2 rounded-xl text-xs font-bold font-head hover:scale-105 transition-all shadow-lg shadow-accent/20"
                                    >
                                        <Play className="w-3 h-3 fill-nav" /> Start Stream
                                    </button>
                                </div>
                            )}

                            <div className="absolute top-4 left-4 flex items-center gap-2">
                                <span className={cn(
                                    "w-2 h-2 rounded-full",
                                    cam.isLive ? "bg-accent animate-pulse" : "bg-neutral-600"
                                )} />
                                <span className="text-[10px] font-mono font-bold text-white uppercase bg-black/40 backdrop-blur-md px-2 py-1 rounded shadow-lg">
                                    {cam.isLive ? 'LIVE' : 'IDLE'}
                                </span>
                            </div>

                            {cam.isLive && (
                                <div className="absolute top-4 right-4 flex items-center gap-2">
                                    <button
                                        onClick={() => toggleCamera(id)}
                                        className="p-2 bg-black/40 backdrop-blur-md rounded-lg text-accent-red hover:bg-accent-red hover:text-white transition-all shadow-lg"
                                    >
                                        <Square className="w-4 h-4 fill-current" />
                                    </button>
                                </div>
                            )}
                        </div>

                        <div className="p-5 border-t border-border bg-white/[0.01]">
                            <div className="flex justify-between items-center mb-4">
                                <div>
                                    <h3 className="text-sm font-semibold text-text">{cam.name}</h3>
                                    <p className="text-[10px] text-muted font-mono mt-1">Source: ID {cam.index}</p>
                                </div>
                                <div className="flex items-center gap-2 bg-background border border-border p-1 rounded-lg">
                                    <select
                                        value={cam.index}
                                        onChange={(e) => updateIndex(id, e.target.value)}
                                        disabled={cam.isLive}
                                        className="bg-transparent text-[10px] text-muted font-mono px-2 py-1 outline-none disabled:opacity-50"
                                    >
                                        {availableSources.map(source => (
                                            <option key={source.id} value={source.id}>
                                                DEVICE {source.id} (Detected)
                                            </option>
                                        ))}
                                        {availableSources.length === 0 && (
                                            <option value={cam.index}>Manual Index {cam.index}</option>
                                        )}
                                    </select>
                                    <Settings className="w-3 h-3 text-muted mr-1" />
                                </div>
                            </div>

                            <div className="w-full bg-white/5 h-[3px] rounded-full overflow-hidden">
                                <div className={cn(
                                    "h-full transition-all duration-1000",
                                    cam.isLive ? "w-full bg-accent" : "w-0 bg-neutral-700"
                                )} />
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default LiveCameras;
