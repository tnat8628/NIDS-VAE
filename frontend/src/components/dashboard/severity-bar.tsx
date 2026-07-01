import { useMemo } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MappedFlow, Severity } from "@/lib/mapper";

interface SeverityBarProps {
  /** Danh sách anomaly đã giới hạn từ view model dự đoán thật. */
  flows: MappedFlow[];
}

const severityLabels: Record<Severity, string> = {
  critical: "nghiêm trọng",
  high: "cao",
  medium: "trung bình",
  low: "thấp",
};

const colors: Record<Severity, string> = {
  critical: "var(--anomaly)",
  high: "var(--warning)",
  medium: "var(--violet)",
  low: "var(--cyan)",
};

const severityOrder: Severity[] = ["critical", "high", "medium", "low"];

export function SeverityBar({ flows }: SeverityBarProps) {
  // Gom nhóm mức độ từ các cảnh báo thật để giữ biểu đồ đồng bộ với AlertTable.
  const severityBuckets = useMemo(
    () =>
      severityOrder.map((severity) => ({
        severity,
        label: severityLabels[severity],
        count: flows.filter((flow) => flow.severity === severity).length,
      })),
    [flows],
  );

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-soft h-full">
      <h3 className="text-sm font-semibold tracking-tight">Mức độ nghiêm trọng</h3>
      <p className="text-xs text-muted-foreground">Mức độ từ lỗi tái tạo của các anomaly mới nhất</p>
      <div className="h-[220px] mt-2">
        <ResponsiveContainer>
          <BarChart data={severityBuckets} layout="vertical" margin={{ left: 8, right: 16 }}>
            <XAxis type="number" hide />
            <YAxis dataKey="label" type="category" axisLine={false} tickLine={false} stroke="var(--muted-foreground)" fontSize={11} width={82} />
            <Tooltip
              contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
              cursor={{ fill: "var(--accent)", opacity: 0.3 }}
            />
            <Bar dataKey="count" radius={[6, 6, 6, 6]}>
              {severityBuckets.map((b) => (
                <Cell key={b.severity} fill={colors[b.severity]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
