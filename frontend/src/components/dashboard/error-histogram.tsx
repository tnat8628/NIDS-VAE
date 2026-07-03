import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistogramBin } from "@/lib/mapper";

interface ErrorHistogramProps {
  /** Histogram aggregate được backend tính từ toàn bộ inference run. */
  histogram: HistogramBin[];
  /** Ngưỡng phân biệt bình thường/bất thường */
  threshold?: number;
}

export function ErrorHistogram({ histogram, threshold }: ErrorHistogramProps) {
  // Tìm bin gần nhất với ngưỡng để vẽ đường tham chiếu
  const thresholdBin =
    threshold === undefined
      ? undefined
      : histogram[Math.max(0, histogram.findIndex((b) => Number(b.bin) >= threshold))]?.bin;

  return (
    <div className="w-full min-w-0 rounded-xl border border-border bg-card p-5 shadow-soft">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Phân bố lỗi tái tạo</h3>
          <p className="text-xs text-muted-foreground">
            {threshold === undefined
              ? "Tổng hợp từ latest inference run của mỗi file"
              : `Lỗi tái tạo theo từng luồng từ VAE · ngưỡng ${threshold.toFixed(4)}`}
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-cyan" /> Bình thường</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-anomaly" /> Bất thường</span>
        </div>
      </div>
      <div className="h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histogram} margin={{ left: -20, right: 8, top: 8, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="bin" stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{
                background: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                fontSize: 12,
              }}
              cursor={{ fill: "var(--accent)", opacity: 0.3 }}
            />
            {thresholdBin && (
              <ReferenceLine
                x={thresholdBin}
                stroke="var(--warning)"
                strokeDasharray="4 4"
                label={{ value: "ngưỡng", fill: "var(--warning)", fontSize: 10, position: "top" }}
              />
            )}
            <Bar dataKey="normal" stackId="a" fill="var(--cyan)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="anomaly" stackId="a" fill="var(--anomaly)" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}