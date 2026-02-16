import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TeamProvider } from './contexts/TeamContext';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Players from './pages/Players';
import Analysis from './pages/Analysis';
import Lineup from './pages/Lineup';
import Settings from './pages/Settings';
import Models from './pages/Models';
import Rankings from './pages/Rankings';
import TeamComparison from './pages/TeamComparison';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TeamProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
            <Route path="players" element={<Players />} />
            <Route path="analysis" element={<Analysis />} />
            <Route path="lineup" element={<Lineup />} />
            <Route path="models" element={<Models />} />
            <Route path="rankings" element={<Rankings />} />
            <Route path="comparison" element={<TeamComparison />} />
            <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </TeamProvider>
    </QueryClientProvider>
  );
}

export default App;
