import React from 'react';

const UnknownFaceQueue = ({ data, loading, count }) => {
    return (
        <div className="bg-surface border border-border rounded-xl p-5 font-body">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-head text-[14px] font-semibold text-text">Unknown Face Queue</h3>
                <span className="text-[10px] bg-accent-red/10 text-accent-red px-2 py-[2px] rounded-full font-mono">
                    {count || 0} pending
                </span>
            </div>

            <div className="space-y-[10px]">
                {loading && <div className="text-center py-6 text-muted text-xs">Loading queue...</div>}
                {!loading && data?.length === 0 && <div className="text-center py-6 text-muted text-xs italic">All clear! No unknown faces.</div>}

                {data?.map((face) => (
                    <div key={face.id} className="bg-accent-red/[0.06] border border-accent-red/[0.15] rounded-xl p-[12px_14px] flex items-center gap-3 animate-in slide-in-from-right duration-300">
                        <div className="w-10 h-10 rounded-lg bg-accent-red/15 flex items-center justify-center text-lg shrink-0 overflow-hidden">
                            {face.image_path ? (
                                <img src={`http://localhost:8000/${face.image_path}`} alt="Unknown" className="w-full h-full object-cover" />
                            ) : "👤"}
                        </div>
                        <div className="flex-1">
                            <p className="text-xs text-text font-medium truncate">Unknown Detected</p>
                            <span className="text-[10px] text-muted font-mono">
                                {new Date(face.timestamp).toLocaleTimeString('en-GB')} · conf {face.confidence?.toFixed(2) || '—'}
                            </span>
                        </div>
                        <button className="bg-accent-red/15 border border-accent-red/25 text-accent-red text-[11px] p-[4px_10px] rounded-md font-mono hover:bg-accent-red/25 transition-all">
                            Review
                        </button>
                    </div>
                ))}
            </div>

            <div className="mt-[14px] border-t border-border pt-[14px]">
                <h4 className="font-head text-[13px] font-semibold text-text mb-[10px]">Suggested Actions</h4>
                <div className="bg-accent-blue/5 border border-border rounded-xl p-3 flex gap-3 cursor-pointer hover:bg-accent-blue/10 transition-all opacity-80">
                    <span className="text-sm">📧</span>
                    <div className="text-[11px] text-muted leading-tight">Generate weekly summary report for head office</div>
                </div>
            </div>
        </div>
    );
};

export default UnknownFaceQueue;
