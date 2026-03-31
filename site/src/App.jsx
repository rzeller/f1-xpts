import { useState } from 'react';
import { useData } from './hooks/useData';
import Nav from './components/Nav';
import Dashboard from './pages/Dashboard';
import Methodology from './pages/Methodology';
import ModelFit from './pages/ModelFit';

export default function App() {
  const { data, loading, error } = useData();
  const [page, setPage] = useState('dashboard');

  if (loading) {
    return (
      <div style={{ padding: '80px 0', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading race data...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '80px 0', textAlign: 'center', color: 'var(--red)' }}>
        Failed to load data: {error.message}
      </div>
    );
  }

  return (
    <>
      <Nav page={page} setPage={setPage} race={data.meta.race} />
      {page === 'dashboard' ? (
        <Dashboard data={data} />
      ) : page === 'methodology' ? (
        <Methodology data={data} />
      ) : (
        <ModelFit data={data} />
      )}
    </>
  );
}
