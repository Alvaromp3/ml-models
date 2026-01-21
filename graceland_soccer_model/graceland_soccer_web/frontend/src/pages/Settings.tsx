import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Calendar, Clock, CheckCircle, RefreshCw, Database } from 'lucide-react';
import { settingsApi } from '../services/api';

export default function Settings() {
  const queryClient = useQueryClient();
  const [saved, setSaved] = useState(false);

  const { data: dateReferenceSetting, isLoading } = useQuery({
    queryKey: ['settings', 'date-reference'],
    queryFn: settingsApi.getDateReference,
  });

  const updateDateReferenceMutation = useMutation({
    mutationFn: (useToday: boolean) => settingsApi.setDateReference(useToday),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
      queryClient.invalidateQueries({ queryKey: ['players'] });
      queryClient.invalidateQueries({ queryKey: ['analysis'] });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    },
  });

  const handleDateReferenceChange = (useToday: boolean) => {
    updateDateReferenceMutation.mutate(useToday);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <div className="p-2 bg-slate-800/60 border border-slate-700/50 rounded-xl">
            <Calendar className="w-5 h-5 text-slate-300" />
          </div>
          Settings
        </h1>
        <p className="text-slate-500 text-sm mt-2 ml-12">
          Configure your dashboard preferences
        </p>
      </div>

      {/* Success message */}
      {saved && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl flex items-center gap-3 animate-slide-in-up">
          <CheckCircle className="w-5 h-5 text-emerald-400" />
          <span className="text-emerald-400 text-sm">Settings saved successfully!</span>
        </div>
      )}

      <div className="max-w-2xl">
        {/* Date Reference Setting */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 bg-slate-800/50 rounded-xl border border-slate-700/50">
              <Calendar className="w-6 h-6 text-slate-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Date Reference</h2>
              <p className="text-sm text-slate-500">Risk calculation period</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <p className="text-sm text-slate-400 mb-4">
                Choose how to calculate the 45-day risk assessment period:
              </p>
              
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-5 h-5 text-slate-400 animate-spin" />
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Option 1: Today's Date */}
                  <label className={`flex items-start gap-4 p-4 rounded-xl border cursor-pointer transition-all hover:bg-slate-800/50 ${
                    dateReferenceSetting?.useTodayAsReference
                      ? 'bg-slate-800/50 border-slate-600/50'
                      : 'bg-slate-800/30 border-slate-700/50'
                  }`}>
                    <input
                      type="radio"
                      name="dateReference"
                      checked={dateReferenceSetting?.useTodayAsReference === true}
                      onChange={() => handleDateReferenceChange(true)}
                      className="mt-1 w-4 h-4 text-slate-400 bg-slate-800 border-slate-700 focus:ring-slate-500 focus:ring-2"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Clock className="w-4 h-4 text-slate-400" />
                        <span className="font-medium text-white">Today's Date</span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Calculate risk from the last 45 days counting from today's date. 
                        Best for real-time monitoring.
                      </p>
                    </div>
                  </label>

                  {/* Option 2: Last Training Date */}
                  <label className={`flex items-start gap-4 p-4 rounded-xl border cursor-pointer transition-all hover:bg-slate-800/50 ${
                    dateReferenceSetting?.useTodayAsReference === false
                      ? 'bg-slate-800/50 border-slate-600/50'
                      : 'bg-slate-800/30 border-slate-700/50'
                  }`}>
                    <input
                      type="radio"
                      name="dateReference"
                      checked={dateReferenceSetting?.useTodayAsReference === false}
                      onChange={() => handleDateReferenceChange(false)}
                      className="mt-1 w-4 h-4 text-slate-400 bg-slate-800 border-slate-700 focus:ring-slate-500 focus:ring-2"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Database className="w-4 h-4 text-slate-400" />
                        <span className="font-medium text-white">Last Training Date</span>
                      </div>
                      <p className="text-xs text-slate-400">
                        Calculate risk from the last 45 days counting from the most recent training session in your CSV. 
                        Best for historical analysis.
                      </p>
                    </div>
                  </label>
                </div>
              )}

              {/* Current setting info */}
              {dateReferenceSetting && (
                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Current setting:</p>
                  <p className="text-sm text-slate-300 font-medium">
                    {dateReferenceSetting.description}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
