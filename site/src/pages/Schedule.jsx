import { useState, useEffect, useMemo } from 'react';
import './Schedule.css';

const SESSION_LABELS = {
  fp1: 'FP1',
  fp2: 'FP2',
  fp3: 'FP3',
  sprint_qualifying: 'SQ',
  sprint: 'Sprint',
  qualifying: 'Quali',
  race: 'Race',
};

const SESSION_ORDER = ['fp1', 'fp2', 'fp3', 'sprint_qualifying', 'sprint', 'qualifying', 'race'];

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function formatPacific(utcStr) {
  const d = new Date(utcStr);
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
  const dayFmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    weekday: 'short',
  });
  const dateFmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    month: 'short',
    day: 'numeric',
  });
  return {
    time: fmt.format(d),
    day: dayFmt.format(d),
    date: dateFmt.format(d),
  };
}

function raceDateRange(sessions) {
  const times = Object.values(sessions).map(s => new Date(s));
  const first = new Date(Math.min(...times));
  const last = new Date(Math.max(...times));

  const fmtStart = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    month: 'short',
    day: 'numeric',
  });
  const fmtEnd = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    month: 'short',
    day: 'numeric',
  });

  const start = fmtStart.format(first);
  const end = fmtEnd.format(last);
  return start === end ? start : `${start} – ${end}`;
}

function RaceCard({ race, isNext, isPast, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);

  const sessions = useMemo(() => {
    return SESSION_ORDER
      .filter(key => race.sessions[key])
      .map(key => ({
        key,
        label: SESSION_LABELS[key],
        ...formatPacific(race.sessions[key]),
        utc: race.sessions[key],
      }));
  }, [race.sessions]);

  const dateRange = useMemo(() => raceDateRange(race.sessions), [race.sessions]);

  return (
    <div className={`race-card${isPast ? ' past' : ''}${isNext ? ' next' : ''}`}>
      <div className="race-card-header" onClick={() => setOpen(!open)}>
        <div className="race-card-left">
          <span className="race-round">R{race.round}</span>
          <span className="race-name">{race.name}</span>
          <span className="race-location">{race.location}</span>
          <div className="race-badges">
            {race.is_sprint && <span className="badge badge-sprint">Sprint</span>}
            {race.saturday_race && <span className="badge badge-saturday">Sat Race</span>}
            {isNext && <span className="badge badge-next">Next</span>}
          </div>
        </div>
        <div className="race-card-right">
          <span className="race-date">{dateRange}</span>
          <span className={`expand-arrow${open ? ' open' : ''}`}>&#9654;</span>
        </div>
      </div>
      {open && (
        <div className="race-card-body">
          <div className="sessions-grid">
            {sessions.map(s => (
              <div className="session-item" key={s.key}>
                <span className={`session-label${s.key === 'sprint_qualifying' ? ' session-label-wide' : ''}`}>
                  {s.label}
                </span>
                <span className="session-day">{s.day}</span>
                <span className="session-time">
                  {s.time}
                  <span className="session-time-local">{s.date}</span>
                </span>
              </div>
            ))}
          </div>
          <div className="circuit-info">
            {race.circuit} &middot; {race.location}, {race.country}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Schedule() {
  const [schedule, setSchedule] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/data/schedule.json')
      .then(r => r.json())
      .then(setSchedule)
      .catch(e => setError(e.message));
  }, []);

  const nextRaceIdx = useMemo(() => {
    if (!schedule) return -1;
    const now = new Date();
    return schedule.races.findIndex(r => new Date(r.sessions.race) > now);
  }, [schedule]);

  if (error) {
    return (
      <div style={{ padding: '80px 0', textAlign: 'center', color: 'var(--red)' }}>
        Failed to load schedule: {error}
      </div>
    );
  }

  if (!schedule) {
    return (
      <div style={{ padding: '80px 0', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading schedule...
      </div>
    );
  }

  return (
    <div className="schedule">
      <div className="schedule-header">
        <h1>2026 Race Schedule</h1>
        <div className="schedule-subtitle">
          All times in Pacific Time &middot; {schedule.races.length} rounds
          &middot; {schedule.races.filter(r => r.is_sprint).length} sprint weekends
        </div>
      </div>
      <div className="race-list">
        {schedule.races.map((race, i) => (
          <RaceCard
            key={race.round}
            race={race}
            isNext={i === nextRaceIdx}
            isPast={i < nextRaceIdx}
            defaultOpen={i === nextRaceIdx}
          />
        ))}
      </div>
    </div>
  );
}
