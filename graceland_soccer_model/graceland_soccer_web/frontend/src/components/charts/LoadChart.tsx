import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { LoadHistory } from '../../types';
import { TrendingUp } from 'lucide-react';

interface LoadChartProps {
  data: LoadHistory[];
}

export default function LoadChart({ data }: LoadChartProps) {
  const avgLoad = data.length > 0 
    ? data.reduce((sum, d) => sum + d.avgLoad, 0) / data.length 
    : 0;

  const formattedData = data
    .filter(d => d.date && d.date !== 'Unknown')
    .map(d => {
      try {
        const date = new Date(d.date);
        if (isNaN(date.getTime())) {
          return null;
        }
        return {
          ...d,
          displayDate: date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric' 
          }),
        };
      } catch {
        return null;
      }
    })
    .filter(d => d !== null);

  return (
    <div 
      className="card p-6 opacity-0 animate-slide-in-up"
      style={{ animationDelay: '150ms', animationFillMode: 'forwards' }}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50">
            <TrendingUp className="w-5 h-5 text-slate-300" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Team Average Load</h3>
            <p className="text-xs text-slate-500">Last 15 Sessions</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500" />
            <span className="text-slate-400">Load</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-emerald-500" style={{ borderStyle: 'dashed' }} />
            <span className="text-slate-400">Avg: {avgLoad.toFixed(0)}</span>
          </div>
        </div>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={formattedData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="loadGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis 
              dataKey="displayDate" 
              stroke="#475569" 
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis 
              stroke="#475569" 
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dx={-10}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(51, 65, 85, 0.5)',
                borderRadius: '12px',
                boxShadow: '0 20px 40px -10px rgba(0, 0, 0, 0.5)',
                backdropFilter: 'blur(10px)',
              }}
              labelStyle={{ color: '#f8fafc', fontWeight: 600, marginBottom: '4px' }}
              formatter={(value) => [`${(value as number).toFixed(1)} units`, 'Avg Load']}
            />
            <ReferenceLine 
              y={avgLoad} 
              stroke="#22c55e" 
              strokeDasharray="6 4" 
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="avgLoad"
              stroke="#06b6d4"
              strokeWidth={2.5}
              fill="url(#loadGradient)"
              dot={{ fill: '#06b6d4', strokeWidth: 0, r: 3 }}
              activeDot={{ 
                fill: '#06b6d4', 
                strokeWidth: 3, 
                stroke: '#0f172a', 
                r: 6,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
