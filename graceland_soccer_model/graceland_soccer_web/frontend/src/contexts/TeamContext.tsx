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
      const response = await fetch('/api/settings/team-status');
      const data = await response.json();
      return data.data;
    },
    refetchInterval: 5000,
  });

  const switchTeamMutation = useMutation({
    mutationFn: async (team: TeamType) => {
      const response = await fetch('/api/settings/switch-team', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ team }),
      });
      const data = await response.json();
      return data.data;
    },
    onSuccess: (data, team) => {
      setCurrentTeam(team);
      queryClient.invalidateQueries();
    },
  });

  useEffect(() => {
    if (teamStatus?.currentTeam) {
      setCurrentTeam(teamStatus.currentTeam);
    }
  }, [teamStatus]);

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
