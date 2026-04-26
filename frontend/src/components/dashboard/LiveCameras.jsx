import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Play, Square, Maximize2, Shield, Activity, Wifi, Settings, RefreshCw, Cpu, Power, Zap, X } from 'lucide-react';
import axios from 'axios';
import { cn } from '../../lib/utils';

const WS_BASE = 'ws://127.0.0.1:8000';

function useCameraWebSocket(canvasRef, camIndex, enabled, fps = 15) {
    const wsRef = useRef(null);
    const lastUrlRef = useRef(null);

    useEffect(() => {
        if (!enabled || camIndex === undefined || camIndex === null) return;
        const canvas = canvasRef?.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d', { alpha: false });
        if (!ctx) return;

        const wsUrl = `${WS_BASE}/ws/cameras/stream?cam_id=${camIndex}&fps=${fps}`;
        const ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';
        wsRef.current = ws;

        ws.onmessage = (evt) => {
            try {
                const bytes = evt.data;
                if (!bytes) return;

                const blob = new Blob([bytes], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);

                const img = new Image();
                img.onload = () => {
                    // Fit image into canvas while keeping aspect ratio
                    const cw = canvas.width;
                    const ch = canvas.height;
                    const iw = img.width;
                    const ih = img.height;
                    if (!cw || !ch || !iw || !ih) {
                        URL.revokeObjectURL(url);
                        return;
                    }

                    const scale = Math.min(cw / iw, ch / ih);
                    const dw = Math.floor(iw * scale);
                    const dh = Math.floor(ih * scale);
                    const dx = Math.floor((cw - dw) / 2);
                    const dy = Math.floor((ch - dh) / 2);

                    ctx.fillStyle = '#000';
                    ctx.fillRect(0, 0, cw, ch);
                    ctx.drawImage(img, dx, dy, dw, dh);

                    URL.revokeObjectURL(url);
                    if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
                    lastUrlRef.current = null;
                };
                img.onerror = () => {
                    URL.revokeObjectURL(url);
                };
                img.src = url;
                lastUrlRef.current = url;
            } catch (e) {
                // ignore
            }
        };

        return () => {
            try {
                ws.close();
            } catch { }
            wsRef.current = null;
            if (lastUrlRef.current) {
                try { URL.revokeObjectURL(lastUrlRef.current); } catch { }
                lastUrlRef.current = null;
            }
        };
    }, [canvasRef, camIndex, enabled, fps]);
}

const LiveCameras = () => {
    const [availableSources, setAvailableSources] = useState([]);
    const [detecting, setDetecting] = useState(false);
    const [engineStatus, setEngineStatus] = useState({ engines: {} });
    const [togglingEngine, setTogglingEngine] = useState(false);
    const [initializingCams, setInitializingCams] = useState({}); // Track which cams are starting
    const [presenceData, setPresenceData] = useState([]);
    const [cameraStates, setCameraStates] = useState({
        'cam-01': { isLive: false, index: 0, name: 'Entrance View', role: 'entry' },
        'cam-02': { isLive: false, index: 1, name: 'Exit View', role: 'exit' },
    });
    const [fullscreenCam, setFullscreenCam] = useState(null);
    const canvasRefs = useRef({});

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

    const fetchPresence = async () => {
        try {
            const resp = await axios.get('http://127.0.0.1:8000/api/students/presence');
            setPresenceData(resp.data);
        } catch (err) {
            console.error("Presence error:", err);
        }
    };

    const fetchEngineStatus = async () => {
        try {
            const resp = await axios.get('http://127.0.0.1:8000/api/engine/status');
            const newStatus = resp.data; // { engines: { [cam_id]: { ... } } }
            setEngineStatus(newStatus);

            // Clean up initializing state for cams that have started
            setInitializingCams(prev => {
                const next = { ...prev };
                Object.keys(newStatus.engines).forEach(cid => {
                    if (newStatus.engines[cid].status === 'running') delete next[cid];
                });
                return next;
            });
        } catch (err) {
            console.error("Engine status error:", err);
        }
    };

    const toggleEngine = async (cam_id, role) => {
        setTogglingEngine(true);
        const currentCamStatus = engineStatus.engines[cam_id]?.status;
        const isRunning = currentCamStatus === 'running';

        if (!isRunning) {
            setInitializingCams(prev => ({ ...prev, [cam_id]: true }));
        }

        try {
            const endpoint = isRunning ? 'stop' : 'start';
            const url = `http://127.0.0.1:8000/api/engine/${endpoint}?cam_id=${cam_id}${endpoint === 'start' ? `&role=${role}` : ''}`;
            await axios.post(url);

            if (isRunning) {
                // Clear initialization state if stopping
                setInitializingCams(prev => {
                    const next = { ...prev };
                    delete next[cam_id];
                    return next;
                });
            }
        } catch (err) {
            console.error("Toggle engine error:", err);
            setInitializingCams(prev => {
                const next = { ...prev };
                delete next[cam_id];
                return next;
            });
        } finally {
            setTogglingEngine(false);
            fetchEngineStatus();
        }
    };

    useEffect(() => {
        detectSources();
        fetchEngineStatus();
        fetchPresence();
        const interval = setInterval(() => {
            fetchEngineStatus();
            fetchPresence();
        }, 3000);
        return () => clearInterval(interval);
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
                <div className="flex gap-2 items-center">
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
                                <CameraCanvas
                                    id={id}
                                    cam={cam}
                                    canvasRefs={canvasRefs}
                                    engineRunning={engineStatus.engines[cam.index]?.status === 'running'}
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
                                <span className="text-[10px] font-mono font-bold text-white uppercase bg-black/40 backdrop-blur-md px-2 py-1 rounded shadow-lg flex items-center gap-2">
                                    {cam.isLive ? 'LIVE' : 'IDLE'}
                                    {cam.isLive && engineStatus.engines[cam.index]?.status === 'running' && (
                                        <span className={cn(
                                            "flex items-center gap-1 border-l border-white/20 pl-2",
                                            engineStatus.engines[cam.index]?.role === 'exit' ? "text-accent-red" : "text-accent"
                                        )}>
                                            <Zap className="w-2 h-2 fill-current" /> AI ACTIVE ({engineStatus.engines[cam.index]?.role.toUpperCase()})
                                        </span>
                                    )}
                                </span>
                            </div>

                            <div className="absolute top-4 right-4 flex items-center gap-2">
                                <button
                                    onClick={() => setFullscreenCam({ id, ...cam })}
                                    className="p-2 bg-black/40 backdrop-blur-md rounded-lg text-white hover:bg-white/20 transition-all shadow-lg"
                                >
                                    <Maximize2 className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={() => toggleCamera(id)}
                                    className="p-2 bg-black/40 backdrop-blur-md rounded-lg text-accent-red hover:bg-accent-red/20 transition-all shadow-lg"
                                >
                                    <Square className="w-4 h-4 fill-current" />
                                </button>
                            </div>
                        </div>

                        <div className="p-5 border-t border-border bg-white/[0.01]">
                            <div className="flex justify-between items-center mb-4">
                                <div>
                                    <h3 className="text-sm font-semibold text-text">{cam.name}</h3>
                                    <p className="text-[10px] text-muted font-mono mt-1">Source: ID {cam.index}</p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="flex items-center gap-2 bg-background border border-border p-1 rounded-lg">
                                        <select
                                            value={cam.index}
                                            onChange={(e) => updateIndex(id, e.target.value)}
                                            disabled={cam.isLive || engineStatus.engines[cam.index]?.status === 'running'}
                                            className="bg-transparent text-[10px] text-muted font-mono px-2 py-1 outline-none disabled:opacity-50"
                                        >
                                            {availableSources.map(source => (
                                                <option key={`src-${source.index}`} value={source.index} className="bg-nav text-text">
                                                    DEVICE {source.index} (Detected)
                                                </option>
                                            ))}
                                            {!availableSources.some(s => s.index == cam.index) && (
                                                <option key={`src-manual-${cam.index}`} value={cam.index} className="bg-nav text-text">DEVICE {cam.index} (Active/Manual)</option>
                                            )}
                                        </select>
                                        <Settings className="w-3 h-3 text-muted mr-1" />
                                    </div>

                                    <button
                                        onClick={() => toggleEngine(cam.index, cam.role)}
                                        disabled={togglingEngine || initializingCams[cam.index] || (engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role !== cam.role)}
                                        className={cn(
                                            "px-3 py-1.5 rounded-lg text-[9px] font-bold font-head transition-all flex items-center gap-2 border shadow-sm relative overflow-hidden",
                                            engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role === cam.role
                                                ? "bg-accent-red/10 border-accent-red/20 text-accent-red hover:bg-accent-red/20"
                                                : "bg-accent/10 border-accent/20 text-accent hover:bg-accent/20",
                                            (initializingCams[cam.index] || (engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role !== cam.role)) && "opacity-50 cursor-not-allowed"
                                        )}
                                    >
                                        {initializingCams[cam.index] && (
                                            <div className="absolute inset-0 bg-accent/10 animate-pulse" />
                                        )}
                                        <Power className={cn("w-2.5 h-2.5 z-10", (togglingEngine || initializingCams[cam.index]) && "animate-pulse")} />
                                        <span className="z-10">
                                            {engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role === cam.role ? 'STOP AI' : `START AI (${cam.role.toUpperCase()})`}
                                        </span>
                                    </button>
                                </div>
                            </div>

                            {engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role !== cam.role && (
                                <div className="mb-3 px-3 py-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg flex items-center gap-2 text-[10px] text-yellow-500 font-mono">
                                    <Shield className="w-3 h-3" />
                                    <span>Device {cam.index} is already assigned to the <b>{engineStatus.engines[cam.index].role.toUpperCase()}</b> tracking engine. Please select a different device.</span>
                                </div>
                            )}

                            <div className="flex items-center justify-between text-[9px] font-mono text-muted/60 mb-2">
                                {engineStatus.engines[cam.index]?.status === 'running' && engineStatus.engines[cam.index]?.role === cam.role ? (
                                    <>
                                        <div className="flex items-center gap-1">
                                            <Activity className="w-2 h-2" /> {engineStatus.engines[cam.index]?.fps} FPS
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Cpu className="w-2 h-2" /> {engineStatus.engines[cam.index]?.gpu_summary}
                                        </div>
                                    </>
                                ) : (
                                    <div className="flex items-center gap-1 italic">
                                        {initializingCams[cam.index] ? 'Initializing...' : 'Recognition Engine Offline'}
                                    </div>
                                )}
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

            {/* Presence Status Table */}
            <div className="bg-surface border border-border rounded-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-700 delay-200">
                <div className="p-4 border-b border-border flex justify-between items-center bg-white/[0.02]">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-accent/10 rounded-lg">
                            <Activity className="w-4 h-4 text-accent" />
                        </div>
                        <h3 className="text-sm font-bold font-head">Presence Analytics (Required: 1.5h)</h3>
                    </div>
                    <div className="flex gap-2">
                        <span className="px-2 py-1 bg-accent/10 text-accent text-[9px] font-bold rounded border border-accent/20">TODAY SESSION</span>
                    </div>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-white/[0.01] text-[10px] font-bold text-muted uppercase tracking-wider">
                                <th className="px-6 py-4 border-b border-border font-mono">Student ID</th>
                                <th className="px-6 py-4 border-b border-border font-mono">Name</th>
                                <th className="px-6 py-4 border-b border-border font-mono">Status</th>
                                <th className="px-6 py-4 border-b border-border font-mono">Entry</th>
                                <th className="px-6 py-4 border-b border-border font-mono">Exit</th>
                                <th className="px-6 py-4 border-b border-border font-mono text-right">Duration</th>
                            </tr>
                        </thead>
                        <tbody className="text-xs">
                            {presenceData.map((row, idx) => (
                                <tr key={row.id} className="hover:bg-white/[0.02] transition-colors group">
                                    <td className="px-6 py-4 border-b border-border/50 font-mono text-muted">{row.id}</td>
                                    <td className="px-6 py-4 border-b border-border/50 font-bold">{row.name}</td>
                                    <td className="px-6 py-4 border-b border-border/50">
                                        <span className={cn(
                                            "px-2 py-1 rounded text-[10px] font-bold uppercase",
                                            row.status === 'present' && "bg-green-500/10 text-green-500 border border-green-500/20",
                                            row.status === 'in_school' && "bg-blue-500/10 text-blue-500 border border-blue-500/20",
                                            row.status === 'under_time' && "bg-orange-500/10 text-orange-500 border border-orange-500/20",
                                            row.status === 'absent' && "bg-muted/10 text-muted border border-muted/20"
                                        )}>
                                            {row.status.replace('_', ' ')}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 border-b border-border/50 font-mono text-muted">{row.entry_time}</td>
                                    <td className="px-6 py-4 border-b border-border/50 font-mono text-muted">{row.exit_time}</td>
                                    <td className="px-6 py-4 border-b border-border/50 text-right font-mono font-bold text-accent">{row.duration}</td>
                                </tr>
                            ))}
                            {presenceData.length === 0 && (
                                <tr>
                                    <td colSpan="6" className="px-6 py-12 text-center text-muted italic border-b border-border">
                                        No student data available for today.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Fullscreen Overlay */}
            {fullscreenCam && (
                <div className="fixed inset-0 z-50 bg-nav/95 backdrop-blur-xl flex flex-col p-8 animate-in fade-in zoom-in duration-300">
                    <div className="flex justify-between items-center mb-6">
                        <div className="flex items-center gap-4">
                            <h2 className="text-xl font-bold font-head">{fullscreenCam.name}</h2>
                            <span className="text-xs font-mono text-muted bg-white/5 px-2 py-1 rounded">Source: ID {fullscreenCam.index}</span>
                        </div>
                        <button
                            onClick={() => setFullscreenCam(null)}
                            className="p-3 bg-white/5 border border-border rounded-xl text-muted hover:text-white hover:bg-white/10 transition-all"
                        >
                            <X className="w-6 h-6" />
                        </button>
                    </div>

                    <div className="flex-1 bg-black rounded-3xl overflow-hidden shadow-2xl border border-border relative">
                        <FullscreenCameraCanvas
                            cam={fullscreenCam}
                            engineRunning={engineStatus.engines[fullscreenCam.index]?.status === 'running'}
                        />

                        <div className="absolute bottom-8 left-8 flex items-center gap-4 bg-black/40 backdrop-blur-md p-4 rounded-2xl border border-white/5">
                            {engineStatus.engines[fullscreenCam.index]?.status === 'running' ? (
                                <>
                                    <div className="flex items-center gap-2 text-accent">
                                        <Activity className="w-4 h-4" />
                                        <span className="font-mono font-bold">{engineStatus.engines[fullscreenCam.index]?.fps} FPS</span>
                                    </div>
                                    <div className="w-px h-4 bg-white/10" />
                                    <div className="flex items-center gap-2 text-muted">
                                        <Cpu className="w-4 h-4" />
                                        <span className="text-xs">{engineStatus.engines[fullscreenCam.index]?.gpu_summary}</span>
                                    </div>
                                </>
                            ) : (
                                <span className="text-xs text-muted italic">Recognition Engine Offline</span>
                            )}
                        </div>

                        {engineStatus.engines[fullscreenCam.index]?.status === 'running' && (
                            <div className={cn(
                                "absolute top-8 left-8 px-4 py-2 border rounded-xl flex items-center gap-2 text-xs font-bold animate-pulse",
                                engineStatus.engines[fullscreenCam.index]?.role === 'exit' ? "bg-accent-red/20 border-accent-red/30 text-accent-red" : "bg-accent/20 border-accent/30 text-accent"
                            )}>
                                <Zap className="w-3 h-3 fill-current" /> AI ANALYTICS ACTIVE ({engineStatus.engines[fullscreenCam.index]?.role.toUpperCase()})
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

const CameraCanvas = ({ id, cam, canvasRefs, engineRunning }) => {
    const canvasRef = useMemo(() => {
        if (!canvasRefs.current[id]) canvasRefs.current[id] = React.createRef();
        return canvasRefs.current[id];
    }, [canvasRefs, id]);

    // Use WebSocket only when engine is running (model running). If engine not running,
    // keep previous MJPEG approach by falling back to <img> (raw feed).
    useCameraWebSocket(canvasRef, cam.index, cam.isLive && engineRunning, 15);

    return engineRunning ? (
        <canvas
            ref={canvasRef}
            className="w-full h-full object-cover"
            width={1280}
            height={720}
        />
    ) : (
        <img
            src={`http://127.0.0.1:8000/api/cameras/stream?cam_id=${cam.index}`}
            className="w-full h-full object-cover"
            alt={cam.name}
            key={`${id}-${cam.isLive}-${cam.index}`}
        />
    );
};

const FullscreenCameraCanvas = ({ cam, engineRunning }) => {
    const canvasRef = useRef(null);
    useCameraWebSocket(canvasRef, cam.index, engineRunning, 20);

    return engineRunning ? (
        <canvas
            ref={canvasRef}
            className="w-full h-full object-contain"
            width={1920}
            height={1080}
        />
    ) : (
        <img
            src={`http://127.0.0.1:8000/api/cameras/stream?cam_id=${cam.index}`}
            className="w-full h-full object-contain"
            alt={cam.name}
        />
    );
};

export default LiveCameras;
