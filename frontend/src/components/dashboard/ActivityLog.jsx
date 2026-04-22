import React from 'react';
import { cn } from '../../lib/utils';

const ActivityLog = ({ data, loading }) => {
    const statusStyles = {
        present: "bg-accent/10 text-accent",
        late: "bg-accent-amber/10 text-accent-amber",
        absent: "bg-accent-red/10 text-accent-red",
        departed: "bg-accent-blue/10 text-accent-blue",
    };

    return (
        <div className="bg-surface border border-border rounded-xl p-5 font-body">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-head text-[14px] font-semibold text-text">Recent Activity</h3>
                <span className="text-[11px] text-accent cursor-pointer font-mono hover:opacity-80">View all →</span>
            </div>

            <div className="space-y-[1px]">
                {loading && <div className="text-center py-10 text-muted text-xs">Loading logs...</div>}
                {!loading && data?.length === 0 && <div className="text-center py-10 text-muted text-xs">No activity yet.</div>}

                {data?.map((log) => (
                    <div key={log.id} className="flex items-center gap-[10px] py-2 border-b border-border last:border-0 text-xs">
                        <span className="font-mono text-[11px] text-muted min-w-[72px]">
                            {new Date(log.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        <span className="text-text font-medium flex-1">Student #{log.student_id}</span>
                        <span className={cn(
                            "text-[10px] px-2 py-[2px] rounded-full font-medium font-mono",
                            statusStyles[log.status] || "bg-white/5 text-muted-2"
                        )}>
                            {log.status}
                        </span>
                        <span className="font-mono text-[10px] text-muted-2 w-8 text-right">
                            {log.confidence_score?.toFixed(2) || '—'}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ActivityLog;
