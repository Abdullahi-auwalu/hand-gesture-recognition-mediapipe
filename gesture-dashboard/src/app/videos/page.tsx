"use client";

import Head from "next/head";
import Link from "next/link";
import useSWR from "swr";
import { getJSON, API } from "@/lib/fetcher";

type VideoItem = {
  file: string;
  bytes: number;
  created_at: string;
  url: string; // relative like "/videos/foo.mp4"
};

export default function VideosPage() {
  const { data, error, isLoading } = useSWR<VideoItem[]>(`/videos`, getJSON);

  return (
    <>
      <Head>
        <title>Recordings • Gesture Dashboard</title>
      </Head>
      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold">Recordings</h1>
          <Link href="/" className="text-sm opacity-80 hover:opacity-100">
            ← Back
          </Link>
        </div>

        {isLoading && <p>Loading…</p>}
        {error && <p className="text-red-500">Failed to load videos.</p>}

        {data && data.length === 0 && (
          <div className="text-sm opacity-70">
            No recordings yet. Press{" "}
            <kbd className="px-1 rounded bg-zinc-700">r</kbd> in the capture
            window to start recording.
          </div>
        )}

        {data && data.length > 0 && (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {data.map((v) => {
              const src = v.url.startsWith("http") ? v.url : `${API}${v.url}`;
              return (
                <div
                  key={v.file}
                  className="rounded-2xl border border-zinc-800 bg-zinc-900/60 overflow-hidden"
                >
                  <video
                    src={src}
                    controls
                    preload="metadata"
                    className="w-full aspect-video"
                  />
                  <div className="p-4">
                    <div className="font-medium">{v.file}</div>
                    <div className="text-xs opacity-70 mt-1">
                      {new Date(v.created_at).toLocaleString()} •{" "}
                      {(v.bytes / 1_048_576).toFixed(1)} MB
                    </div>
                    <div className="mt-3">
                      <a
                        className="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700"
                        href={src}
                        download
                      >
                        Download
                      </a>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </>
  );
}

// import { API, getJSON } from "@/lib/fetcher";
// import { Video } from "@/types";
// import Link from "next/link";

// async function loadVideos(): Promise<Video[]> {
//   return getJSON("/videos");
// }

// export default async function VideosPage() {
//   const data = await loadVideos();

//   return (
//     <main className="max-w-6xl mx-auto px-4 py-6">
//       <div className="flex items-center justify-between mb-6">
//         <h1 className="text-2xl font-semibold">Recordings</h1>
//         <Link href="/" className="text-sm opacity-80 hover:opacity-100">
//           ← Back
//         </Link>
//       </div>

//       {data.length === 0 ? (
//         <div className="text-sm opacity-70">
//           No recordings yet. Press{" "}
//           <kbd className="px-1 rounded bg-zinc-700">r</kbd> in the capture
//           window to start recording.
//         </div>
//       ) : (
//         <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
//           {data.map((v) => {
//             const src = v.url.startsWith("http") ? v.url : `${API}${v.url}`;
//             return (
//               <div
//                 key={v.file}
//                 className="rounded-2xl border border-zinc-800 bg-zinc-900/60 overflow-hidden"
//               >
//                 <video
//                   src={src}
//                   controls
//                   preload="metadata"
//                   className="w-full aspect-video"
//                 />
//                 <div className="p-4">
//                   <div className="font-medium break-all">{v.file}</div>
//                   <div className="text-xs opacity-70 mt-1">
//                     {new Date(v.created_at).toLocaleString()} •{" "}
//                     {(v.bytes / 1_048_576).toFixed(1)} MB
//                   </div>
//                   <div className="mt-3">
//                     <a
//                       className="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700"
//                       href={src}
//                       download
//                     >
//                       Download
//                     </a>
//                   </div>
//                 </div>
//               </div>
//             );
//           })}
//         </div>
//       )}
//     </main>
//   );
// }
