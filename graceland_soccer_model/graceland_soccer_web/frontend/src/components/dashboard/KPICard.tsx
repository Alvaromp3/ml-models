import { TrendingUp, TrendingDown } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string | number;
  change: number;
  icon: LucideIcon;
  subtitle?: string;
  variant?: 'default' | 'warning' | 'success';
  delay?: number;
}

export default function KPICard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  subtitle,
  variant = 'default',
  delay = 0
}: KPICardProps) {
  const isPositive = change >= 0;
  
  const config = {
    default: {
      gradient: 'from-cyan-500 to-blue-600',
      shadow: 'shadow-cyan-500/20',
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/20',
    },
    warning: {
      gradient: 'from-orange-500 to-red-600',
      shadow: 'shadow-red-500/20',
      bg: 'bg-red-500/10',
      border: 'border-red-500/20',
    },
    success: {
      gradient: 'from-emerald-500 to-green-600',
      shadow: 'shadow-emerald-500/20',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
    },
  };

  const { gradient, shadow, bg, border } = config[variant];

  return (
    <div 
      className={`
        card card-hover p-6 opacity-0 animate-slide-in-up
      `}
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient} ${shadow} shadow-lg`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        
        <div className={`
          flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold
          ${isPositive 
            ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' 
            : 'bg-red-500/15 text-red-400 border border-red-500/20'
          }
        `}>
          {isPositive ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
          <span>{Math.abs(change)}%</span>
        </div>
      </div>
      
      <div>
        <h3 className="text-3xl font-bold text-white mb-1 count-up">{value}</h3>
        <p className="text-sm font-medium text-slate-400">{title}</p>
        {subtitle && (
          <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
        )}
      </div>

      {/* Bottom accent line */}
      <div className={`mt-5 h-1 rounded-full bg-gradient-to-r ${gradient} opacity-40`} />
    </div>
  );
}
