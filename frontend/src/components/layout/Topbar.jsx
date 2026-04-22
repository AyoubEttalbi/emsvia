import React, { useState, useEffect } from 'react';

const Topbar = ({ title }) => {
    const [time, setTime] = useState(new Date().toLocaleTimeString('en-GB'));

    useEffect(() => {
        const timer = setInterval(() => {
            setTime(new Date().toLocaleTimeString('en-GB'));
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="h-[56px] border-b border-border flex items-center px-7 gap-4 shrink-0 font-body">
            <div className="font-head text-[16px] font-semibold flex-1 text-text">{title}</div>
            <span className="bg-accent/10 border border-accent/20 text-accent text-[11px] px-[10px] py-[3px] rounded-full font-mono font-medium">
                ● Live
            </span>
            <span className="font-mono text-[12px] text-muted tracking-[0.05em]">
                {time}
            </span>
        </div>
    );
};

export default Topbar;
