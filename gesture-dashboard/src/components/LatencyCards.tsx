export default function LatencyCards({
  latency,
}: {
  latency: Record<string, number | null>;
}) {
  const fmt = (v: number | null) => (v == null ? "—" : `${v.toFixed(1)} ms`);
  const box = (label: string, value: string) => (
    <div
      style={{
        padding: 12,
        border: "1px solid #eee",
        borderRadius: 8,
        minWidth: 120,
      }}
    >
      <div style={{ opacity: 0.7, fontSize: 12 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600 }}>{value}</div>
    </div>
  );
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
      {box("Mean", fmt(latency.mean))}
      {box("p50", fmt(latency.p50))}
      {box("p90", fmt(latency.p90))}
      {box("p99", fmt(latency.p99))}
    </div>
  );
}
