import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '../services/api';

type TeamType = 'mens' | 'womens';

interface TeamContextType {
  currentTeam: TeamType;
  switchTeam: (team: TeamType) => void;
  teamStatus: {
    mens: { loaded: boolean; rowCount: number };
    womens: { loaded: boolean; rowCount: number };
  } | null;
  isLoading: boolean;
}

const TeamContext = createContext<TeamContextType | undefined>(undefined);

export function TeamProvider({ children }: { children: ReactNode }) {
  const [currentTeam, setCurrentTeam] = useState<TeamType>('mens');
  const queryClient = useQueryClient();

  const { data: teamStatus, isLoading } = useQuery({
    queryKey: ['teamStatus'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/settings/team-status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (!data.success) {
          throw new Error(data.error || 'Failed to fetch team status');
        }
        return data.data;
      } catch (error) {
        console.error('Error fetching team status:', error);
        // Return default structure on error
        return {
          currentTeam: 'mens',
          mens: { loaded: false, rowCount: 0 },
          womens: { loaded: false, rowCount: 0 }
        };
      }
    },
    refetchInterval: 5000,
    retry: 2,
    retryDelay: 1000,
  });

  const switchTeamMutation = useMutation({
    mutationFn: async (team: TeamType) => {
      try {
        const response = await fetch('/api/settings/switch-team', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ team }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (!data.success) {
          throw new Error(data.error || 'Failed to switch team');
        }
        return data.data;
      } catch (error) {
        console.error('Error switching team:', error);
        throw error;
      }
    },
    onSuccess: (data, team) => {
      setCurrentTeam(team);
      queryClient.invalidateQueries();
    },
  });

  // Sync currentTeam from server only on initial load (when we don't have a pending switch).
  // Do NOT force the user back to the team that has data — they must be able to switch
  // to the other tab (e.g. Women's) to upload that team's CSV when only Men's is loaded.
  useEffect(() => {
    if (!teamStatus?.currentTeam) return;
    setCurrentTeam(teamStatus.currentTeam);
  }, [teamStatus?.currentTeam]);

  const switchTeam = (team: TeamType) => {
    switchTeamMutation.mutate(team);
  };

  return (
    <TeamContext.Provider
      value={{
        currentTeam,
        switchTeam,
        teamStatus: teamStatus || null,
        isLoading,
      }}
    >
      {children}
    </TeamContext.Provider>
  );
}

export function useTeam() {
  const context = useContext(TeamContext);
  if (context === undefined) {
    throw new Error('useTeam must be used within a TeamProvider');
  }
  return context;
}
