import React, { useState, useEffect, useRef } from 'react';
import { getLogs } from '../../api/client';
import { Terminal, Copy, Trash2, RefreshCw } from 'lucide-react';

const LogViewer = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const scrollRef = useRef(null);

    const fetchLogs = async () => {
        try {
            const resp = await getLogs();
            setLogs(resp.data.logs);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLogs();
        const interval = setInterval(fetchLogs, 3000); // 3s polling
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="bg-surface border border-border rounded-2xl overflow-hidden flex flex-col h-[400px]">
            <div className="p-4 border-b border-border bg-white/2 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-accent" />
                    <span className="text-xs font-mono font-bold text-text uppercase tracking-widest">Inference Engine Logs</span>
                </div>
                <div className="flex gap-2">
                    <button onClick={fetchLogs} className="p-1.5 hover:bg-white/5 rounded text-muted transition-colors">
                        <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                    <button className="p-1.5 hover:bg-white/5 rounded text-muted transition-colors">
                        <Copy className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-5 font-mono text-[11px] leading-relaxed space-y-1.5 scrollbar-thin scrollbar-thumb-white/10"
            >
                {loading && logs.length === 0 ? (
                    <div className="text-muted opacity-50 italic">Establishing connection to log stream...</div>
                ) : (
                    logs.map((line, i) => {
                        const isError = line.includes('ERROR') || line.includes('Critical');
                        const isWarn = line.includes('WARNING');
                        const isInfo = line.includes('INFO');

                        return (
                            <div key={i} className="flex gap-3 group">
                                <span className="text-muted/30 select-none w-8 text-right">{i + 1}</span>
                                <span className={cn(
                                    "flex-1 break-all",
                                    isError && "text-accent-red",
                                    isWarn && "text-yellow-500/80",
                                    isInfo && "text-accent/80",
                                    !isError && !isWarn && !isInfo && "text-muted-2"
                                )}>
                                    {line}
                                </span>
                            </div>
                        );
                    })
                )}
            </div>

            <div className="p-3 border-t border-border bg-white/[0.01] flex justify-between items-center px-5">
                <div className="flex items-center gap-4 text-[9px] text-muted font-mono uppercase">
                    <span className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                        Live Stream
                    </span>
                    <span>Lines: {logs.length}</span>
                </div>
                <div className="text-[9px] text-muted-2 font-mono italic">
                    tail -n 50 emsvia.log
                </div>
            </div>
        </div>
    );
};

// Internal utility since we're in one file
function cn(...classes) {
    return classes.filter(Boolean).join(' ');
}

export default LogViewer;
