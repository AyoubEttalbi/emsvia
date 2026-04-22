import React from 'react';
import { cn } from '../../lib/utils';

const MetricCard = ({ label, value, delta, icon, color = 'green', trend = 'up' }) => {
    const colorClasses = {
        green: "before:bg-accent hover:border-accent/40",
        blue: "before:bg-accent-blue hover:border-accent-blue/40",
        red: "before:bg-accent-red hover:border-accent-red/40",
        amber: "before:bg-accent-amber hover:border-accent-amber/40",
    };

    return (
        <div className={cn(
            "bg-surface border border-border rounded-xl p-[18px_20px] relative overflow-hidden transition-all duration-200 before:content-[''] before:absolute before:top-0 before:left-0 before:right-0 before:h-[2px] before:rounded-[12px_12px_0_0]",
            colorClasses[color]
        )}>
            <div className="text-[11px] text-muted tracking-[0.08em] uppercase mb-2 font-mono">{label}</div>
            <div className="font-head text-[28px] font-bold text-text mb-1 leading-none">{value}</div>
            <div className={cn(
                "text-[11px] font-medium mt-[6px]",
                trend === 'up' ? "text-accent" : "text-accent-red"
            )}>
                {trend === 'up' ? '▲' : '▼'} {delta}
            </div>
            <div className="absolute top-[14px] right-4 opacity-20 text-[28px]">
                {icon}
            </div>
        </div>
    );
};

export default MetricCard;
