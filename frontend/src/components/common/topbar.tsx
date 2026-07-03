import { Bell, ShieldAlert } from "lucide-react";

export function Topbar({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="h-16 border-b border-border bg-background/60 backdrop-blur sticky top-0 z-30">
      <div className="h-full px-6 flex items-center justify-between gap-4 max-w-[1280px] mx-auto w-full">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold tracking-tight truncate">{title}</h1>
          {subtitle && <p className="text-xs text-muted-foreground truncate">{subtitle}</p>}
        </div>

        <div className="flex items-center gap-3">
          <button className="relative h-9 w-9 grid place-items-center rounded-lg border border-border bg-muted/40 hover:bg-muted transition">
            <Bell className="h-4 w-4" />
            <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-anomaly" />
          </button>
          <div className="flex items-center gap-2 pl-3 border-l border-border">
            <div className="h-8 w-8 rounded-full bg-gradient-primary grid place-items-center text-xs font-semibold text-primary-foreground">SA</div>
            <div className="hidden md:block leading-tight">
              <div className="text-sm font-medium">Chuyên viên SOC</div>
              <div className="text-[10px] text-muted-foreground flex items-center gap-1"><ShieldAlert className="h-3 w-3" /> Cấp 2</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}