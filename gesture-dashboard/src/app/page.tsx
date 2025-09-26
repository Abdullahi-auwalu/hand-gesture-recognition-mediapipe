"use client";
import useSWR from "swr";
import Link from "next/link";
import { getJSON } from "../lib/fetcher";
import { BarChart3, RefreshCcw } from "lucide-react";

type Run = {
  id: string;
  file: string;
  bytes: number;
  started_at?: string | null;
};

export default function Home() {
  const {
    data: runs,
    isLoading,
    error,
    mutate,
  } = useSWR<Run[]>("/runs", getJSON, { refreshInterval: 2500 });

  return (
    <div className="grid lg:grid-cols-[360px_1fr] gap-6">
      {/* Runs list */}
      <section className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="h2">Runs</div>
          <button onClick={() => mutate()} className="button">
            <RefreshCcw size={16} /> Refresh
          </button>
        </div>

        {isLoading && <div className="text-mute">Loading…</div>}
        {error && <div className="text-bad">Failed to load /runs</div>}
        {!isLoading && (!runs || runs.length === 0) && (
          <div className="text-mute text-sm">
            No runs yet. Start your app and make sure it writes{" "}
            <span className="kbd">logs/run_*.jsonl</span>.
          </div>
        )}

        <ul className="space-y-2 mt-2">
          {runs?.map((r) => (
            <li key={r.id}>
              <Link
                href={`/run/${encodeURIComponent(r.id)}`}
                className="block rounded-xl border border-border hover:border-brand/40 hover:bg-panel/70 transition p-3"
              >
                <div className="flex items-center justify-between">
                  <div className="font-medium">{r.id}</div>
                  <span className="text-xs text-mute">
                    {(r.bytes / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div className="text-xs text-mute mt-1">
                  {r.started_at || r.file}
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Empty preview */}
      <section className="card p-6 grid place-items-center min-h-[420px]">
        <div className="text-center opacity-80">
          <BarChart3 className="mx-auto mb-4" />
          <div className="text-lg font-medium">Metrics</div>
          <div className="text-sm text-mute">
            Select a run from the left to view metrics.
          </div>
        </div>
      </section>
    </div>
  );
}
