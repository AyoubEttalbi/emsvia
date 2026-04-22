import React, { useState } from 'react';
import { X, Upload, Check, Loader2 } from 'lucide-react';
import axios from 'axios';
import { cn } from '../../lib/utils';

const EnrollmentModal = ({ isOpen, onClose, onSuccess }) => {
    const [loading, setLoading] = useState(false);
    const [files, setFiles] = useState([]);
    const [formData, setFormData] = useState({
        student_id: '',
        first_name: '',
        last_name: '',
        email: '',
        phone: ''
    });

    if (!isOpen) return null;

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        const data = new FormData();
        Object.keys(formData).forEach(key => data.append(key, formData[key]));
        files.forEach(file => data.append('files', file));

        try {
            await axios.post('http://127.0.0.1:8000/api/students/', data, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            onSuccess();
            onClose();
        } catch (err) {
            alert(err.response?.data?.detail || "Enrollment failed");
        } finally {
            setLoading(false);
        }
    };

    const handleFileChange = (e) => {
        setFiles([...files, ...Array.from(e.target.files)]);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

            <div className="relative bg-surface border border-border w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
                <div className="p-6 border-b border-border flex justify-between items-center bg-white/2">
                    <div>
                        <h2 className="font-head text-lg font-bold text-text">New Enrollment</h2>
                        <p className="text-xs text-muted font-mono uppercase mt-1">Register Student & Biosamples</p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-lg transition-colors text-muted">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <label className="text-[10px] text-muted font-mono uppercase tracking-wider">Student ID</label>
                            <input
                                required
                                className="w-full bg-background border border-border rounded-lg p-2 text-sm focus:border-accent outline-none font-mono"
                                placeholder="EMS-2024-001"
                                value={formData.student_id}
                                onChange={e => setFormData({ ...formData, student_id: e.target.value })}
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] text-muted font-mono uppercase tracking-wider">Email</label>
                            <input
                                type="email"
                                className="w-full bg-background border border-border rounded-lg p-2 text-sm focus:border-accent outline-none font-mono"
                                placeholder="name@university.edu"
                                value={formData.email}
                                onChange={e => setFormData({ ...formData, email: e.target.value })}
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <label className="text-[10px] text-muted font-mono uppercase tracking-wider">First Name</label>
                            <input
                                required
                                className="w-full bg-background border border-border rounded-lg p-2 text-sm focus:border-accent outline-none font-body"
                                placeholder="John"
                                value={formData.first_name}
                                onChange={e => setFormData({ ...formData, first_name: e.target.value })}
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] text-muted font-mono uppercase tracking-wider">Last Name</label>
                            <input
                                required
                                className="w-full bg-background border border-border rounded-lg p-2 text-sm focus:border-accent outline-none font-body"
                                placeholder="Doe"
                                value={formData.last_name}
                                onChange={e => setFormData({ ...formData, last_name: e.target.value })}
                            />
                        </div>
                    </div>

                    <div className="space-y-3">
                        <label className="text-[10px] text-muted font-mono uppercase tracking-wider">Face Samples (Images)</label>
                        <div
                            className={cn(
                                "border-2 border-dashed border-border rounded-xl p-8 flex flex-col items-center justify-center transition-all cursor-pointer hover:border-accent/50 hover:bg-accent/[0.02]",
                                files.length > 0 && "border-accent/40 bg-accent/[0.01]"
                            )}
                            onClick={() => document.getElementById('file-upload').click()}
                        >
                            <input id="file-upload" type="file" multiple hidden onChange={handleFileChange} accept="image/*" />
                            {files.length === 0 ? (
                                <>
                                    <Upload className="w-8 h-8 text-muted mb-2 opacity-50" />
                                    <p className="text-xs text-muted-2">Drop images or click to browse</p>
                                    <p className="text-[10px] text-muted mt-1 font-mono uppercase">Min 3 photos recommended</p>
                                </>
                            ) : (
                                <div className="grid grid-cols-4 gap-2">
                                    {files.map((f, i) => (
                                        <div key={i} className="w-12 h-12 rounded-lg bg-accent/20 flex items-center justify-center text-accent">
                                            <X className="w-3 h-3 absolute -top-1 -right-1 bg-accent-red rounded-full text-white cursor-pointer" onClick={(e) => {
                                                e.stopPropagation();
                                                setFiles(files.filter((_, idx) => idx !== i));
                                            }} />
                                            🖼️
                                        </div>
                                    ))}
                                    <div className="w-12 h-12 rounded-lg border border-dashed border-accent/40 flex items-center justify-center text-accent">
                                        +
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading || files.length === 0}
                        className="w-full bg-accent text-nav font-head font-bold p-3 rounded-xl hover:opacity-90 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Check className="w-5 h-5" />}
                        {loading ? 'Processing Samples...' : 'Complete Enrollment'}
                    </button>

                    <p className="text-[9px] text-muted-2 text-center font-mono uppercase tracking-widest px-4 leading-relaxed">
                        Upon completion, the AI engine will automatically generate face embeddings for the provided samples.
                    </p>
                </form>
            </div>
        </div>
    );
};

export default EnrollmentModal;
