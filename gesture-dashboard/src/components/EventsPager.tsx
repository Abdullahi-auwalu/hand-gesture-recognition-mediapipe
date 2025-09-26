import useSWR from "swr";
import { getJSON } from "../lib/fetcher";
import { PagedEvents } from "../types";
import { useState } from "react";

export default function EventsPager({ runId }: { runId: string }) {
  const [offset, setOffset] = useState(0);
  const limit = 50;

  const { data, error, isLoading } = useSWR<PagedEvents>(
    `/runs/${encodeURIComponent(runId)}/events?offset=${offset}&limit=${limit}`,
    getJSON
  );

  if (isLoading) return <p>Loading events…</p>;
  if (error)
    return (
      <p style={{ color: "crimson" }}>Failed to load events: {String(error)}</p>
    );
  if (!data) return null;

  const canPrev = offset > 0;
  const canNext = offset + limit < data.total;

  return (
    <div>
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          margin: "8px 0",
        }}
      >
        <button
          onClick={() => setOffset(Math.max(0, offset - limit))}
          disabled={!canPrev}
        >
          Prev
        </button>
        <span>
          Showing {offset + 1}–{Math.min(offset + limit, data.total)} of{" "}
          {data.total}
        </span>
        <button onClick={() => setOffset(offset + limit)} disabled={!canNext}>
          Next
        </button>
      </div>

      <pre
        style={{
          maxHeight: 300,
          overflow: "auto",
          background: "#0b0b0b",
          color: "#d6d6d6",
          padding: 12,
          borderRadius: 8,
        }}
      >
        {JSON.stringify(data.events, null, 2)}
      </pre>
    </div>
  );
}
