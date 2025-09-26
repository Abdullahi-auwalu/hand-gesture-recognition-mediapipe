import "./globals.css";
import { ReactNode } from "react";
import { Activity, Github } from "lucide-react";
import SidebarNav from "@/components/SidebarNav";
// import SidebarNav from "./_components/SidebarNav";

export const metadata = {
  title: "Gesture Dashboard",
  description: "Realtime gesture analysis",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="grid grid-cols-[280px_1fr] min-h-screen">
          {/* Sidebar */}
          <aside className="sticky top-0 h-screen p-4 border-r border-border bg-panel">
            <div className="flex items-center gap-3 mb-6">
              <div className="size-9 rounded-xl bg-brand/20 border border-brand/30 grid place-items-center">
                <Activity className="text-brand" size={18} />
              </div>
              <div>
                <div className="font-semibold">Gesture Runs</div>
                <div className="text-xs text-mute">analysis &amp; insights</div>
              </div>
            </div>

            <SidebarNav />

            <div className="absolute bottom-4 left-4 right-4 text-xs text-mute flex items-center justify-between">
              <div className="opacity-70">⌘K quick search soon</div>
              <a
                className="opacity-60 hover:opacity-100"
                href="#"
                aria-label="repo"
              >
                <Github size={16} />
              </a>
            </div>
          </aside>

          {/* Main column */}
          <div className="min-h-screen">
            {/* Topbar */}
            <header className="sticky top-0 z-10 bg-bg/70 backdrop-blur border-b border-border">
              <div className="flex items-center justify-between px-6 py-3">
                <div className="h1">Dashboard</div>
                <div className="flex items-center gap-3">
                  <span className="hidden md:inline text-mute text-sm">
                    API
                  </span>
                  <span className="rounded-xl bg-good/15 text-good border border-good/20 px-2 py-1 text-xs">
                    Connected
                  </span>
                </div>
              </div>
            </header>

            <main className="p-6">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}

// import "./globals.css";
// import { ReactNode } from "react";
// import Link from "next/link";
// import { Activity, Github } from "lucide-react";

// export const metadata = {
//   title: "Gesture Dashboard",
//   description: "Realtime gesture analysis",
// };

// export default function RootLayout({ children }: { children: ReactNode }) {
//   return (
//     <html lang="en">
//       <body>
//         <div className="grid grid-cols-[280px_1fr] min-h-screen">
//           {/* Sidebar */}
//           <aside className="sticky top-0 h-screen p-4 border-r border-border bg-panel">
//             <div className="flex items-center gap-3 mb-6">
//               <div className="size-9 rounded-xl bg-brand/20 border border-brand/30 grid place-items-center">
//                 <Activity className="text-brand" size={18} />
//               </div>
//               <div>
//                 <div className="font-semibold">Gesture Runs</div>
//                 <div className="text-xs text-mute">analysis & insights</div>
//               </div>
//             </div>

//             <nav className="space-y-1">
//               <Link className="button w-full justify-center" href="/">
//                 All Runs
//               </Link>
//               <a
//                 className="button w-full justify-center"
//                 href="http://127.0.0.1:8000/docs"
//                 target="_blank"
//                 rel="noreferrer"
//               >
//                 API Docs
//               </a>
//             </nav>

//             <div className="absolute bottom-4 left-4 right-4 text-xs text-mute flex items-center justify-between">
//               <div className="opacity-70">⌘K quick search soon</div>
//               <a
//                 className="opacity-60 hover:opacity-100"
//                 href="#"
//                 aria-label="repo"
//               >
//                 <Github size={16} />
//               </a>
//             </div>
//           </aside>

//           {/* Main column */}
//           <div className="min-h-screen">
//             {/* Topbar */}
//             <header className="sticky top-0 z-10 bg-bg/70 backdrop-blur border-b border-border">
//               <div className="flex items-center justify-between px-6 py-3">
//                 <div className="h1">Dashboard</div>
//                 <div className="flex items-center gap-3">
//                   <span className="hidden md:inline text-mute text-sm">
//                     API
//                   </span>
//                   <span className="rounded-xl bg-good/15 text-good border border-good/20 px-2 py-1 text-xs">
//                     Connected
//                   </span>
//                 </div>
//               </div>
//             </header>

//             <main className="p-6">{children}</main>
//           </div>
//         </div>
//       </body>
//     </html>
//   );
// }
