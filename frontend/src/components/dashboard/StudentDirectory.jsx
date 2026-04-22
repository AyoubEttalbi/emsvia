import React, { useState, useEffect } from 'react';
import { getStudents } from '../../api/client';
import { Search, UserPlus, Filter, MoreVertical } from 'lucide-react';
import { cn } from '../../lib/utils';
import EnrollmentModal from './EnrollmentModal';

const StudentDirectory = () => {
    const [students, setStudents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);

    const fetchStudents = async () => {
        try {
            const response = await getStudents();
            setStudents(response.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStudents();
    }, []);

    const filteredStudents = students.filter(s =>
        s.first_name.toLowerCase().includes(search.toLowerCase()) ||
        s.last_name.toLowerCase().includes(search.toLowerCase()) ||
        s.student_id.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="flex justify-between items-center bg-surface border border-border p-4 rounded-xl">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                    <input
                        type="text"
                        placeholder="Search by name or ID..."
                        className="w-full bg-background border border-border rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-accent transition-all font-body"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                </div>
                <div className="flex gap-3">
                    <button className="flex items-center gap-2 bg-white/5 border border-border px-4 py-2 rounded-lg text-sm text-text hover:bg-white/10 transition-all font-mono">
                        <Filter className="w-4 h-4" /> Filter
                    </button>
                    <button
                        onClick={() => setIsModalOpen(true)}
                        className="flex items-center gap-2 bg-accent text-nav px-4 py-2 rounded-lg text-sm font-bold hover:opacity-90 transition-all font-head"
                    >
                        <UserPlus className="w-4 h-4" /> Add Student
                    </button>
                </div>
            </div>

            <div className="bg-surface border border-border rounded-xl overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-white/2 border-b border-border text-[10px] text-muted uppercase font-mono">
                            <th className="px-6 py-4 font-semibold tracking-wider">Student</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Student ID</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Email</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Enrollment Date</th>
                            <th className="px-6 py-4 font-semibold tracking-wider">Status</th>
                            <th className="px-6 py-4 font-semibold tracking-wider"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {loading && (
                            <tr>
                                <td colSpan="6" className="px-6 py-10 text-center text-muted font-mono text-sm leading-6">
                                    Fetching directory records...
                                </td>
                            </tr>
                        )}
                        {!loading && filteredStudents.length === 0 && (
                            <tr>
                                <td colSpan="6" className="px-6 py-10 text-center text-muted font-mono text-sm leading-6">
                                    No students found matching your criteria.
                                </td>
                            </tr>
                        )}
                        {filteredStudents.map((student) => (
                            <tr key={student.id} className="hover:bg-white/[0.02] transition-colors group">
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-9 h-9 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center text-accent text-sm font-bold font-mono">
                                            {student.first_name[0]}{student.last_name[0]}
                                        </div>
                                        <div>
                                            <div className="text-sm font-medium text-text">{student.first_name} {student.last_name}</div>
                                            <div className="text-[11px] text-muted font-mono">{student.is_active ? 'Active' : 'Inactive'}</div>
                                        </div>
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-xs font-mono text-muted-2 uppercase">{student.student_id}</td>
                                <td className="px-6 py-4 text-xs font-mono text-muted-2">{student.email}</td>
                                <td className="px-6 py-4 text-xs font-mono text-muted-2">
                                    {new Date(student.enrollment_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-4">
                                    <span className={cn(
                                        "text-[10px] px-2 py-[2px] rounded-full font-medium font-mono uppercase tracking-tighter",
                                        student.is_active ? "bg-accent/10 text-accent" : "bg-accent-red/10 text-accent-red"
                                    )}>
                                        {student.is_active ? 'Active' : 'Archived'}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-right">
                                    <button className="text-muted hover:text-text transition-colors">
                                        <MoreVertical className="w-4 h-4" />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <EnrollmentModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSuccess={fetchStudents}
            />
        </div>
    );
};

export default StudentDirectory;
