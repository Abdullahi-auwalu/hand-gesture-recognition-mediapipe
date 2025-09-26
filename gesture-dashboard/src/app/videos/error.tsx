"use client";
export default function Error({ error }: { error: Error }) {
  return (
    <div className="max-w-6xl mx-auto px-4 py-6 text-red-500">
      Failed to load videos: {error.message}
    </div>
  );
}
