import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Upload,
  ListChecks,
  BarChart3,
  ShieldCheck,
  Settings,
} from "lucide-react";

const nav = [
  { label: "Tổng quan", to: "/dashboard", icon: LayoutDashboard },
  { label: "Tải lên & Phân tích", to: "/upload", icon: Upload },
  { label: "Kết quả", to: "/results", icon: ListChecks },
  // { label: "Trạng thái hệ thống", to: "/health", icon: Activity },
  { label: "Phân tích", to: "/analytics", icon: BarChart3 },
  // { label: "Quản lý mô hình", to: "/models", icon: Cpu },
] as const;

export function AppSidebar() {
  const { pathname } = useLocation();

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
      <Link to="/" className="flex items-center gap-2.5 px-5 h-16 border-b border-sidebar-border">
        <div className="relative h-9 w-9 rounded-xl bg-gradient-primary grid place-items-center shadow-glow">
          <ShieldCheck className="h-5 w-5 text-primary-foreground" />
        </div>
        <div className="leading-tight">
          <div className="font-semibold text-sm tracking-tight">VAE NIDS</div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Bảo mật AI</div>
        </div>
      </Link>

      <nav className="flex-1 px-3 py-4 space-y-1">
        <div className="px-2 pb-2 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          Không gian làm việc
        </div>

        {nav.map((item) => {
          const active =
            pathname === item.to ||
            (item.to !== "/dashboard" && pathname.startsWith(item.to));

          const Icon = item.icon;

          return (
            <Link
              key={item.to}
              to={item.to}
              className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all ${
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-soft"
                  : "text-sidebar-foreground/75 hover:text-sidebar-foreground hover:bg-sidebar-accent/60"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-gradient-primary" />
              )}
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="px-3 pb-4">
        <Link
          to="/dashboard"
          className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-sidebar-foreground/60 hover:bg-sidebar-accent/60"
        >
          <Settings className="h-4 w-4" /> Cài đặt
        </Link>

        <div className="mt-3 rounded-xl border border-sidebar-border bg-sidebar-accent/40 p-3">
          <div className="flex items-center gap-2 text-xs">
            <span className="relative inline-flex h-2 w-2 rounded-full bg-success pulse-dot text-success" />
            <span className="text-muted-foreground">Mô hình hoạt động</span>
          </div>
          <div className="mt-1 text-xs font-mono text-foreground/80">
            vae-nids-v1.4.2
          </div>
        </div>
      </div>
    </aside>
  );
}

export default AppSidebar;
