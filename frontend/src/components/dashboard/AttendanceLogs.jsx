import React, { useState, useEffect } from 'react';
import { getAttendance } from '../../api/client';
import { Calendar, Search, Download, Filter } from 'lucide-react';
import { cn } from '../../lib/utils';

const AttendanceLogs = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetch = async () => {
            try {
                const response = await getAttendance();
                setLogs(response.data);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetch();
    }, []);

    const statusStyles = {
        present: "bg-accent/10 text-accent",
        late: "bg-accent-amber/10 text-accent-amber",
        absent: "bg-accent-red/10 text-accent-red",
        departed: "bg-accent-blue/10 text-accent-blue",
    };

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="flex justify-between items-center bg-surface border border-border p-4 rounded-xl">
                <div className="flex gap-4">
                    <div className="flex items-center gap-2 bg-background border border-border rounded-lg px-3 py-2 text-xs text-muted font-mono">
                        <Calendar className="w-4 h-4" />
                        <span>Today: {new Date().toLocaleDateString()}</span>
                    </div>
                    <div className="flex items-center gap-2 bg-background border border-border rounded-lg px-3 py-2 text-xs text-muted font-mono">
                        <Filter className="w-4 h-4" />
                        <span>All Entrances</span>
                    </div>
                </div>
                <button className="flex items-center gap-2 bg-accent-blue/20 text-accent-blue border border-accent-blue/30 px-4 py-2 rounded-lg text-sm font-semibold hover:bg-accent-blue/30 transition-all font-head">
                    <Download className="w-4 h-4" /> Export CSV
                </button>
            </div>

            <div className="bg-surface border border-border rounded-xl font-body overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-white/2 border-b border-border text-[10px] text-muted uppercase font-mono">
                            <th className="px-6 py-4 font-semibold tracking-wider">Timestamp</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Student ID</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Status</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Confidence</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Camera ID</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Snapshot</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {loading && (
                            <tr>
                                <td colSpan="6" className="px-6 py-10 text-center text-muted font-mono text-xs">
                                    Retrieving system logs...
                                </td>
                            </tr>
                        )}
                        {logs.map((log) => (
                            <tr key={log.id} className="hover:bg-white/[0.02] transition-colors">
                                <td className="px-6 py-4">
                                    <div className="text-xs text-text font-medium">{new Date(log.timestamp).toLocaleTimeString()}</div>
                                    <div className="text-[10px] text-muted font-mono">{new Date(log.timestamp).toLocaleDateString()}</div>
                                </td>
                                <td className="px-6 py-4 text-xs font-mono text-text">#{log.student_id}</td>
                                <td className="px-6 py-4">
                                    <span className={cn(
                                        "text-[10px] px-2 py-[2px] rounded-full font-medium font-mono uppercase",
                                        statusStyles[log.status] || "bg-white/5 text-muted-2"
                                    )}>
                                        {log.status}
                                    </span>
                                </td>
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-2">
                                        <div className="w-12 h-[2px] bg-white/10 rounded-full overflow-hidden">
                                            <div className="h-full bg-accent" style={{ width: `${log.confidence_score * 100}%` }} />
                                        </div>
                                        <span className="text-[10px] text-muted-2 font-mono">{(log.confidence_score * 100).toFixed(0)}%</span>
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-xs font-mono text-muted-2">{log.camera_id}</td>
                                <td className="px-6 py-4">
                                    {log.image_path ? (
                                        <div className="w-8 h-8 rounded border border-border bg-black/20 overflow-hidden cursor-pointer hover:border-accent transition-all ring-accent hover:ring-1">
                                            {/* Image preview link or component */}
                                            <div className="w-full h-full flex items-center justify-center text-[10px] bg-accent/20">📷</div>
                                        </div>
                                    ) : <span className="text-[10px] text-muted italic">No image</span>}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default AttendanceLogs;
