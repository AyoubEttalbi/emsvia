import React from 'react';
import { LayoutGrid, Users, CalendarDays, Eye, Camera, Activity, Settings, Cpu } from 'lucide-react';
import { cn } from '../../lib/utils';
import { useStats } from '../../hooks/useStats';

const Sidebar = ({ activePage, setActivePage }) => {
    const { stats } = useStats();

    const menuItems = [
        { id: 'overview', label: 'Overview', icon: LayoutGrid, section: 'Core' },
        { id: 'students', label: 'Students', icon: Users, section: 'Core' },
        { id: 'attendance', label: 'Attendance', icon: CalendarDays, section: 'Core' },
        { id: 'review', label: 'Face Review', icon: Eye, section: 'Security', badge: stats?.unknown_faces_pending },
        { id: 'cameras', label: 'Live Cameras', icon: Camera, section: 'Security' },
        { id: 'health', label: 'System Health', icon: Activity, section: 'System' },
        { id: 'settings', label: 'Settings', icon: Settings, section: 'System' },
    ];

    return (
        <nav className="w-[220px] min-w-[220px] bg-nav border-r border-border flex flex-col shrink-0 h-screen sticky top-0 font-body">
            <div className="p-[24px_20px_20px] border-b border-border">
                <div className="w-9 h-9 bg-accent rounded-xl flex items-center justify-center mb-[10px]">
                    <Cpu className="w-5 h-5 text-nav fill-nav" />
                </div>
                <h1 className="font-head text-[15px] font-bold text-text tracking-[0.05em]">EMSVIA</h1>
                <p className="text-[10px] text-muted mt-[2px] tracking-[0.08em] uppercase">Biometric Attendance</p>
            </div>

            <div className="flex-1 overflow-y-auto">
                {['Core', 'Security', 'System'].map((section) => (
                    <div key={section} className="mb-4">
                        <div className="p-[16px_12px_8px] text-[10px] text-muted tracking-[0.12em] uppercase font-mono">
                            {section}
                        </div>
                        {menuItems.filter(item => item.section === section).map((item) => (
                            <div
                                key={item.id}
                                onClick={() => setActivePage(item.id)}
                                className={cn(
                                    "flex items-center gap-[10px] p-[9px_16px] m-[1px_8px] rounded-lg cursor-pointer transition-all duration-150 text-[13px] font-medium",
                                    activePage === item.id
                                        ? "bg-accent/10 text-accent font-semibold"
                                        : "text-muted-2 hover:bg-white/5 hover:text-text"
                                )}
                            >
                                <item.icon className={cn("w-4 h-4 shrink-0", activePage === item.id ? "opacity-100" : "opacity-70")} />
                                {item.label}
                                {item.badge > 0 && (
                                    <span className="ml-auto bg-accent-red text-white text-[10px] font-semibold p-[1px_6px] rounded-xl font-mono">
                                        {item.badge}
                                    </span>
                                )}
                            </div>
                        ))}
                    </div>
                ))}
            </div>

            <div className="mt-auto p-4 border-t border-border">
                <div className="bg-accent-blue/10 border border-accent-blue/20 rounded-lg p-[10px_12px]">
                    <div className="flex items-center">
                        <span className="w-[6px] h-[6px] rounded-full bg-accent mr-[6px] animate-pulse" />
                        <span className="text-[11px] text-accent-blue font-mono font-medium">GPU Active</span>
                    </div>
                    <div className="text-[10px] text-muted mt-[3px]">GTX 1050 · CUDA 12.1</div>
                </div>
            </div>
        </nav>
    );
};

export default Sidebar;
