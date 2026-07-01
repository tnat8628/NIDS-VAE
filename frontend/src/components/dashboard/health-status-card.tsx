import { CircleDot } from "lucide-react";

export interface HealthStatusItem {
  label: string;
  status: "online" | "degraded" | "offline";
  uptime: string;
  latency: string;
}

interface HealthStatusCardsProps {
  /** Dữ liệu trạng thái được truyền từ page để component không phụ thuộc mockData. */
  items: HealthStatusItem[];
}

const tone: Record<string, string> = {
  online: "text-success",
  degraded: "text-warning",
  offline: "text-anomaly",
};

const statusLabel: Record<string, string> = {
  online: "Hoạt động",
  degraded: "Suy giảm",
  offline: "Ngoại tuyến",
};

export function HealthStatusCards({ items }: HealthStatusCardsProps) {
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {items.map((h) => (
        <div key={h.label} className="rounded-xl border border-border bg-card p-5 shadow-soft relative overflow-hidden">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">{h.label}</div>
              <div className={`mt-2 flex items-center gap-2 text-sm font-medium capitalize ${tone[h.status]}`}>
                <span className="relative">
                  <CircleDot className="h-3.5 w-3.5" />
                  {h.status === "online" && <span className="absolute inset-0 h-3.5 w-3.5 rounded-full text-success pulse-dot" />}
                </span>
                {statusLabel[h.status] ?? h.status}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Thời gian hoạt động</div>
              <div className="font-mono text-sm">{h.uptime}</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-border flex justify-between text-xs">
            <span className="text-muted-foreground">Độ trễ</span>
            <span className="font-mono">{h.latency}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
