import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { RiskDistribution } from '../../types';

interface RiskDonutProps {
  data: RiskDistribution;
}

const COLORS = {
  low: '#22c55e',
  medium: '#eab308', 
  high: '#ef4444',
};

export default function RiskDonut({ data }: RiskDonutProps) {
  const total = data.low + data.medium + data.high;
  
  const chartData = [
    { name: 'Low Risk', value: data.low, color: COLORS.low },
    { name: 'Medium Risk', value: data.medium, color: COLORS.medium },
    { name: 'High Risk', value: data.high, color: COLORS.high },
  ].filter(d => d.value > 0);

  // If no data, show empty state
  if (chartData.length === 0) {
    chartData.push({ name: 'No Data', value: 1, color: '#334155' });
  }

  return (
    <div className="h-44 relative">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={70}
            paddingAngle={chartData.length > 1 ? 4 : 0}
            dataKey="value"
            stroke="none"
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={entry.color}
                className="transition-all duration-300 hover:opacity-80"
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(51, 65, 85, 0.5)',
              borderRadius: '12px',
              padding: '8px 12px',
            }}
            formatter={(value) => [`${value} players`, '']}
          />
        </PieChart>
      </ResponsiveContainer>
      
      {/* Center text */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="text-center">
          <p className="text-3xl font-bold text-white">{total}</p>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Players</p>
        </div>
      </div>
    </div>
  );
}
