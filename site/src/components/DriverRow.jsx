import Sparkline from './Sparkline';
import BarGauge from './BarGauge';

function fmt(n, decimals = 1) {
  return n.toFixed(decimals);
}

function pct(n) {
  return (n * 100).toFixed(1) + '%';
}

export default function DriverRow({ driver, rank, teamColor, maxEP, minEP, isSprint }) {
  const ep = driver.ep_total;
  const isNeg = ep < 0;

  return (
    <tr className="driver-row">
      <td className="col-rank">{rank}</td>
      <td className="col-driver">
        <span className="team-dot" style={{ background: teamColor }} />
        <span className="driver-name">{driver.name}</span>
        <span className="driver-abbr">{driver.abbr}</span>
      </td>
      <td className="col-sparkline">
        <Sparkline distribution={driver.position_distribution} color={teamColor} />
      </td>
      <td className="col-ep">
        <div className="ep-cell">
          <div className="ep-values">
            <span className={isNeg ? 'negative' : ''}>{fmt(ep)}</span>
            {isSprint && driver.ep_sprint > 0 && (
              <span className="ep-breakdown">
                {fmt(driver.ep_race)}+{fmt(driver.ep_sprint)}
              </span>
            )}
          </div>
          <BarGauge value={ep} maxEP={maxEP} minEP={minEP} color={teamColor} />
        </div>
      </td>
      <td className="col-pct">{pct(driver.p_win)}</td>
      <td className="col-pct">{pct(driver.p_podium)}</td>
      <td className="col-pct">{pct(driver.p_top10)}</td>
      <td className="col-pct">{pct(driver.p_dnf)}</td>
      <td className="col-num muted">{fmt(driver.std_dev)}</td>
    </tr>
  );
}
