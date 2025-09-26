"use client";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import Link from "next/link";
import { getJSON } from "../../../lib/fetcher";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { ArrowLeft } from "lucide-react";

type Metrics = {
  run_id: string;
  total_frames: number;
  class_histogram: Record<string, number>;
  confusion: Record<string, Record<string, number>>;
  latency: { mean?: number; p50?: number; p90?: number; p99?: number };
  labels_seen: string[];
};

const COLORS = [
  "#6ee7ff",
  "#58d38c",
  "#ffc857",
  "#ff6b6b",
  "#c099ff",
  "#f59e0b",
  "#22c55e",
];

function Stat({
  label,
  value,
  tone = "",
}: {
  label: string;
  value: string;
  tone?: "good" | "bad" | "";
}) {
  const toneCls =
    tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-text";
  return (
    <div className="card p-4">
      <div className="text-xs text-mute">{label}</div>
      <div className={`text-2xl mt-2 font-semibold ${toneCls}`}>{value}</div>
    </div>
  );
}

export default function RunPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { data, error, isLoading } = useSWR<Metrics>(
    `/runs/${encodeURIComponent(id)}/metrics`,
    getJSON,
    { refreshInterval: 2500 }
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button className="button" onClick={() => router.push("/")}>
          <ArrowLeft size={16} /> Back
        </button>
        <div className="h1">Run {id}</div>
      </div>

      {isLoading && <div className="text-mute">Loading metrics…</div>}
      {error && <div className="text-bad">Failed to load metrics.</div>}
      {!isLoading && data && (
        <>
          {/* Stats row */}
          <div className="grid md:grid-cols-4 gap-4">
            <Stat
              label="Total frames"
              value={data.total_frames.toLocaleString()}
            />
            <Stat
              label="Mean latency (ms)"
              value={(data.latency.mean ?? 0).toFixed(2)}
            />
            <Stat label="p90 (ms)" value={(data.latency.p90 ?? 0).toFixed(2)} />
            <Stat
              label="p99 (ms)"
              value={(data.latency.p99 ?? 0).toFixed(2)}
              tone="bad"
            />
          </div>

          {/* Histogram + Latency */}
          <div className="grid lg:grid-cols-2 gap-6">
            <section className="card p-4">
              <div className="h2 mb-3">Class histogram</div>
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Object.entries(data.class_histogram).map(
                      ([k, v]) => ({ label: k, value: v })
                    )}
                  >
                    <CartesianGrid stroke="#1d232d" />
                    <XAxis dataKey="label" stroke="#a0a6b2" />
                    <YAxis stroke="#a0a6b2" />
                    <Tooltip
                      contentStyle={{
                        background: "#0b0e12",
                        border: "1px solid #1d232d",
                      }}
                    />
                    <Bar dataKey="value">
                      {Object.keys(data.class_histogram).map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="card p-4">
              <div className="h2 mb-3">Latency distribution</div>
              <div className="grid md:grid-cols-3 gap-3 mb-4">
                <Stat label="p50" value={(data.latency.p50 ?? 0).toFixed(2)} />
                <Stat label="p90" value={(data.latency.p90 ?? 0).toFixed(2)} />
                <Stat
                  label="p99"
                  value={(data.latency.p99 ?? 0).toFixed(2)}
                  tone="bad"
                />
              </div>
              <div className="h-[240px] grid place-items-center text-mute text-sm">
                (Optional) Add a histogram chart fed by latency samples.
              </div>
            </section>
          </div>

          {/* Confusion matrix */}
          <section className="card p-4">
            <div className="h2 mb-3">Confusion matrix</div>
            <div className="overflow-auto">
              <table className="text-sm border-separate border-spacing-0">
                <thead>
                  <tr>
                    <th className="sticky left-0 z-10 bg-panel/95 p-2 border-b border-border text-mute">
                      GT \ Pred
                    </th>
                    {data.labels_seen.map((l) => (
                      <th
                        key={l}
                        className="px-3 py-2 border-b border-border text-mute"
                      >
                        {l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.labels_seen.map((gt) => (
                    <tr key={gt}>
                      <th className="sticky left-0 z-10 bg-panel/95 p-2 border-r border-border text-mute">
                        {gt}
                      </th>
                      {data.labels_seen.map((pred, i) => {
                        const v = data.confusion?.[gt]?.[pred] ?? 0;
                        const tone =
                          gt === pred ? "bg-good/20" : v > 0 ? "bg-bad/20" : "";
                        return (
                          <td
                            key={pred}
                            className={`px-3 py-2 text-center border-b border-border ${tone}`}
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
          </section>

          {/* Raw export */}
          <section className="card p-4">
            <div className="flex items-center justify-between">
              <div className="h2">Raw events</div>
              <Link
                className="button"
                href={`/api/runs/${encodeURIComponent(id)}/export.csv`}
                target="_blank"
              >
                Download CSV
              </Link>
            </div>
            <div className="text-sm text-mute mt-2">
              A paginated event viewer can go here (infinite scroll with
              windowing).
            </div>
          </section>
        </>
      )}
    </div>
  );
}
