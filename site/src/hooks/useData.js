import { useState, useEffect, useCallback, useRef } from 'react';

export function useData() {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const cache = useRef({});

  // Load race index on mount
  useEffect(() => {
    fetch('/data/races/index.json')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(index => {
        setRaces(index.races || []);
        setSelectedRace(index.latest || null);
      })
      .catch(() => {
        // Fallback: no index yet, load latest.json directly
        setSelectedRace('__latest__');
      });
  }, []);

  // Load race data when selection changes
  useEffect(() => {
    if (!selectedRace) return;

    // Check cache
    if (cache.current[selectedRace]) {
      setData(cache.current[selectedRace]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    const url = selectedRace === '__latest__'
      ? '/data/latest.json'
      : `/data/races/${selectedRace}.json`;

    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        cache.current[selectedRace] = d;
        setData(d);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }, [selectedRace]);

  return { data, loading, error, races, selectedRace, setSelectedRace };
}
