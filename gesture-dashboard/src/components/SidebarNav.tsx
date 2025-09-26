"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
// import { cn } from "@/lib/utils"; // or replace cn(...) with template strings if you don't have a helper

const items = [
  { href: "/", label: "All Runs" },
  { href: "/videos", label: "Recordings" },
  { href: "http://127.0.0.1:8000/docs", label: "API Docs", external: true },
];

export default function SidebarNav() {
  const pathname = usePathname();

  return (
    <nav className="space-y-1">
      {items.map((it) =>
        it.external ? (
          <a
            key={it.label}
            className="button w-full justify-center"
            href={it.href}
            target="_blank"
            rel="noreferrer"
          >
            {it.label}
          </a>
        ) : (
          <Link
            key={it.href}
            href={it.href}
            className={`button w-full justify-center ${
              pathname === it.href ? "bg-zinc-900/60 border-zinc-700" : ""
            }`}
          >
            {it.label}
          </Link>
        )
      )}
    </nav>
  );
}
