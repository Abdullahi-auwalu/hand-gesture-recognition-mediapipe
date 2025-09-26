import useSWR from "swr";
import Link from "next/link";
import { getJSON } from "../lib/fetcher";
import { Run } from "@/types";

export default function RunList() {
  const { data, error, isLoading } = useSWR<Run[]>("/runs", getJSON);

  if (isLoading) return <p>Loading runs…</p>;
  if (error)
    return (
      <p style={{ color: "crimson" }}>Failed to load runs: {String(error)}</p>
    );
  if (!data || data.length === 0)
    return <p>No runs yet. Start your app to generate logs.</p>;

  return (
    <ul style={{ listStyle: "none", padding: 0 }}>
      {data.map((r) => (
        <li
          key={r.id}
          style={{
            margin: "12px 0",
            padding: "12px",
            border: "1px solid #eee",
            borderRadius: 8,
          }}
        >
          <div>
            <b>Run:</b> {r.id} &nbsp;{" "}
            <small>({(r.bytes / 1024).toFixed(1)} KB)</small>
          </div>
          <div>
            <small>{r.started_at ?? "unknown start"}</small>
          </div>
          <div style={{ marginTop: 8 }}>
            <Link href={`/run/${r.id}`}>Open metrics</Link>
            &nbsp;|&nbsp;
            <a
              href={`/api/runs/${encodeURIComponent(r.id)}/export.csv`}
              target="_blank"
              rel="noreferrer"
            >
              CSV
            </a>
            &nbsp;|&nbsp;
            <a
              href={`/api/runs/${encodeURIComponent(r.id)}/raw`}
              target="_blank"
              rel="noreferrer"
            >
              Raw
            </a>
          </div>
        </li>
      ))}
    </ul>
  );
}
