import React, { useState } from 'react';
import { useUnknowns } from '../../hooks/useUnknowns';
import { reviewUnknown, dismissUnknown } from '../../api/client';
import { Check, X, ShieldAlert, Cpu } from 'lucide-react';
import { cn } from '../../lib/utils';

const FaceReview = () => {
    const { unknowns, loading, refetch } = useUnknowns();
    const [processing, setProcessing] = useState(null);

    const handleAction = async (id, action) => {
        setProcessing(id);
        try {
            if (action === 'dismiss') await dismissUnknown(id);
            else await reviewUnknown(id);

            // We assume useUnknowns will eventually provide a refetch
            // For now, let's just wait it out or hope the interval catches it
            if (refetch) refetch();
        } catch (err) {
            console.error(err);
        } finally {
            setProcessing(null);
        }
    };

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="bg-accent-red/[0.04] border border-accent-red/20 p-6 rounded-2xl flex items-center gap-6">
                <div className="w-16 h-16 rounded-2xl bg-accent-red/15 flex items-center justify-center text-accent-red">
                    <ShieldAlert className="w-8 h-8" />
                </div>
                <div>
                    <h2 className="font-head text-lg font-bold text-text mb-1">Security Audit Hub</h2>
                    <p className="text-sm text-muted-2 max-w-xl leading-relaxed">
                        Review unrecognized faces captured by the GPU-accelerated engine. Dismissing an entry removes it from the pending queue, while marking it reviewed keeps the record.
                    </p>
                </div>
                <div className="ml-auto text-right">
                    <div className="text-3xl font-head font-bold text-accent-red leading-none">{unknowns.length}</div>
                    <div className="text-[10px] text-muted font-mono uppercase tracking-widest mt-2">Pending faces</div>
                </div>
            </div>

            {loading && (
                <div className="flex flex-col items-center justify-center py-20 text-muted font-mono text-sm leading-10">
                    <Cpu className="w-8 h-8 animate-spin mb-4 opacity-40 text-accent" />
                    Analyzing recognition buffers...
                </div>
            )}

            {!loading && unknowns.length === 0 && (
                <div className="text-center py-24 bg-surface border border-border rounded-2xl border-dashed">
                    <div className="text-4xl mb-4 text-accent/40">💎</div>
                    <h3 className="text-text font-head font-bold">All Secure</h3>
                    <p className="text-muted text-sm mt-2 font-body">No unrecognized faces found in the database.</p>
                </div>
            )}

            <div className="grid grid-cols-3 gap-5">
                {unknowns.map((face) => (
                    <div key={face.id} className={cn(
                        "bg-surface border border-border rounded-2xl overflow-hidden group hover:border-accent-red/30 transition-all duration-300",
                        processing === face.id && "opacity-50 pointer-events-none"
                    )}>
                        <div className="aspect-[4/3] bg-black relative">
                            {face.image_path ? (
                                <img
                                    src={`http://127.0.0.1:8000/${face.image_path}`}
                                    className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                                    alt="Captured Unknown"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center text-muted text-xs">No preview available</div>
                            )}
                            <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md px-2 py-1 rounded-lg text-[10px] font-mono text-accent-red border border-accent-red/20">
                                {face.confidence?.toFixed(2)} CONF
                            </div>
                        </div>
                        <div className="p-5 space-y-4">
                            <div>
                                <div className="text-xs text-muted font-mono mb-1">{new Date(face.timestamp).toLocaleString()}</div>
                                <div className="text-sm font-semibold text-text">Unknown Face detected at Entry A</div>
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleAction(face.id, 'dismiss')}
                                    className="flex-1 flex items-center justify-center gap-2 bg-white/5 border border-border p-2 rounded-xl text-xs text-muted-2 hover:bg-accent-red/10 hover:text-accent-red hover:border-accent-red/20 transition-all font-mono"
                                >
                                    <X className="w-3 h-3" /> Dismiss
                                </button>
                                <button
                                    onClick={() => handleAction(face.id, 'review')}
                                    className="flex-1 flex items-center justify-center gap-2 bg-accent/10 border border-accent/20 p-2 rounded-xl text-xs text-accent hover:bg-accent hover:text-nav transition-all font-mono font-bold"
                                >
                                    <Check className="w-3 h-3" /> Mark Reviewed
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default FaceReview;
