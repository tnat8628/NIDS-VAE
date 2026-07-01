import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { MappedSummary } from "@/lib/mapper";

interface AnomalyDonutProps {
  /** Tóm tắt từ lần dự đoán VAE mới nhất trong localStorage. */
  summary: MappedSummary;
}

export function AnomalyDonut({ summary }: AnomalyDonutProps) {
  // Dựng dữ liệu biểu đồ từ kết quả thật, không dùng số liệu mock.
  const data = [
    { name: "Bình thường", value: summary.normalCount, fill: "var(--cyan)" },
    { name: "Bất thường", value: summary.anomalyCount, fill: "var(--anomaly)" },
  ];

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-soft h-full">
      <h3 className="text-sm font-semibold tracking-tight">Phân loại luồng</h3>
      <p className="text-xs text-muted-foreground">Batch phân tích hiện tại</p>
      <div className="h-[220px] relative">
        <ResponsiveContainer>
          <PieChart>
            <Pie data={data} dataKey="value" innerRadius={62} outerRadius={88} paddingAngle={3} stroke="none">
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                fontSize: 12,
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <div className="text-2xl font-semibold font-mono">{summary.anomalyRatePercent.toFixed(1)}%</div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Tỷ lệ bất thường</div>
        </div>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg border border-border p-2">
          <div className="text-muted-foreground">Bình thường</div>
          <div className="font-mono text-base">{summary.normalCount.toLocaleString()}</div>
        </div>
        <div className="rounded-lg border border-border p-2">
          <div className="text-muted-foreground">Bất thường</div>
          <div className="font-mono text-base text-anomaly">{summary.anomalyCount.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}
