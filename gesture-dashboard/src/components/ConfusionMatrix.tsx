type Conf = Record<string, Record<string, number>>;

export default function ConfusionMatrix({
  conf,
  labels,
}: {
  conf: Conf;
  labels: string[];
}) {
  if (!labels || labels.length === 0) return <p>No labels.</p>;
  const get = (gt: string, pred: string) => conf?.[gt]?.[pred] ?? 0;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", minWidth: 600 }}>
        <thead>
          <tr>
            <th
              style={{
                border: "1px solid #eee",
                padding: "6px 8px",
                background: "#fafafa",
              }}
            >
              GT \\ Pred
            </th>
            {labels.map((l) => (
              <th
                key={l}
                style={{
                  border: "1px solid #eee",
                  padding: "6px 8px",
                  background: "#fafafa",
                }}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((gt) => (
            <tr key={gt}>
              <th
                style={{
                  border: "1px solid #eee",
                  padding: "6px 8px",
                  textAlign: "left",
                  background: "#fafafa",
                }}
              >
                {gt}
              </th>
              {labels.map((pred) => {
                const v = get(gt, pred);
                const bg =
                  v > 0
                    ? `rgba(0, 128, 255, ${Math.min(0.1 + v / 30, 0.9)})`
                    : "transparent";
                return (
                  <td
                    key={pred}
                    style={{
                      border: "1px solid #eee",
                      padding: "6px 8px",
                      textAlign: "center",
                      background: bg,
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
