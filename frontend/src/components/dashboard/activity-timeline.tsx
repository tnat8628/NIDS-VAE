import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MappedFlow } from "@/lib/mapper";

interface ActivityTimelineProps {
  /** Preview flow đã được mapper giới hạn để biểu đồ không xử lý payload lớn. */
  flows: MappedFlow[];
}

export function ActivityTimeline({ flows }: ActivityTimelineProps) {
  // Backend chưa trả timestamp, nên timeline biểu diễn các đoạn liên tiếp theo thứ tự row.
  const timeline = useMemo(() => {
    if (flows.length === 0) return [];

    const bucketCount = Math.min(12, Math.max(1, flows.length));
    const bucketSize = Math.ceil(flows.length / bucketCount);

    return Array.from({ length: bucketCount }, (_, index) => {
      const bucketFlows = flows.slice(index * bucketSize, (index + 1) * bucketSize);
      return {
        hour: `#${index + 1}`,
        normal: bucketFlows.filter((flow) => flow.prediction === 0).length,
        anomaly: bucketFlows.filter((flow) => flow.prediction === 1).length,
      };
    });
  }, [flows]);

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-soft">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Diễn biến batch</h3>
          <p className="text-xs text-muted-foreground">Luồng theo các đoạn row trong lần phân tích mới nhất</p>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-cyan" /> Bình thường</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-anomaly" /> Bất thường</span>
        </div>
      </div>
      <div className="h-[240px]">
        <ResponsiveContainer>
          <AreaChart data={timeline} margin={{ left: -20, right: 8 }}>
            <defs>
              <linearGradient id="normalG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--cyan)" stopOpacity={0.4} />
                <stop offset="100%" stopColor="var(--cyan)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="anomalyG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--anomaly)" stopOpacity={0.5} />
                <stop offset="100%" stopColor="var(--anomaly)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="hour" stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
            />
            <Area type="monotone" dataKey="normal" stroke="var(--cyan)" strokeWidth={2} fill="url(#normalG)" />
            <Area type="monotone" dataKey="anomaly" stroke="var(--anomaly)" strokeWidth={2} fill="url(#anomalyG)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
