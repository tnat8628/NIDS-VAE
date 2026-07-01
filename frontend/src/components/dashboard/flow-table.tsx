import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Download, Search } from "lucide-react";
import type { MappedFlow } from "@/lib/mapper";

interface FlowTableProps {
  /** Danh sach flow da duoc cat gioi han thanh preview nhe RAM. */
  flows: MappedFlow[];
  /** Bao hieu bang chi dang hien thi ban xem truoc. */
  isPreview?: boolean;
  /** Tong so flow goc tu backend, dung de hien thi canh bao preview. */
  totalRows?: number;
}

const sevTone: Record<string, string> = {
  critical: "text-anomaly bg-anomaly/10 border-anomaly/30",
  high: "text-warning bg-warning/10 border-warning/30",
  medium: "text-violet bg-violet/10 border-violet/30",
  low: "text-cyan bg-cyan/10 border-cyan/30",
};

const sevLabel: Record<string, string> = {
  critical: "nghiêm trọng",
  high: "cao",
  medium: "trung bình",
  low: "thấp",
};

export function FlowTable({ flows, isPreview = false, totalRows }: FlowTableProps) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<"all" | "anomaly" | "normal">("all");
  const [sort, setSort] = useState<"err-desc" | "err-asc" | "idx">("err-desc");
  const [page, setPage] = useState(0);
  const pageSize = 25;

  // Chi loc/sap xep tren mang preview nho; khong bao gio xu ly hang trieu row trong component.
  const filtered = useMemo(() => {
    let nextFlows = flows;
    if (filter === "anomaly") nextFlows = nextFlows.filter((flow) => flow.prediction === 1);
    if (filter === "normal") nextFlows = nextFlows.filter((flow) => flow.prediction === 0);
    if (q) nextFlows = nextFlows.filter((flow) => String(flow.rowIndex).includes(q));
    if (sort === "err-desc") return [...nextFlows].sort((a, b) => b.reconstructionError - a.reconstructionError);
    if (sort === "err-asc") return [...nextFlows].sort((a, b) => a.reconstructionError - b.reconstructionError);
    return [...nextFlows].sort((a, b) => a.rowIndex - b.rowIndex);
  }, [q, filter, sort, flows]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const view = filtered.slice(page * pageSize, page * pageSize + pageSize);

  return (
    <div className="rounded-xl border border-border bg-card shadow-soft">
      <div className="flex flex-wrap items-center gap-3 p-4 border-b border-border">
        <div className="flex items-center gap-2 px-3 h-9 rounded-lg border border-border bg-muted/40 text-sm flex-1 min-w-[200px]">
          <Search className="h-4 w-4 text-muted-foreground" />
          <input
            value={q}
            onChange={(event) => { setQ(event.target.value); setPage(0); }}
            placeholder="Tìm theo chỉ số hàng..."
            className="bg-transparent outline-none flex-1 text-sm"
          />
        </div>
        <div className="flex rounded-lg border border-border overflow-hidden text-xs">
          {(["all", "anomaly", "normal"] as const).map((nextFilter) => (
            <button
              key={nextFilter}
              onClick={() => { setFilter(nextFilter); setPage(0); }}
              className={`px-3 h-9 capitalize transition ${filter === nextFilter ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              {nextFilter === "all" ? "Tất cả" : nextFilter === "anomaly" ? "Bất thường" : "Bình thường"}
            </button>
          ))}
        </div>
        <select
          value={sort}
          onChange={(event) => setSort(event.target.value as typeof sort)}
          className="h-9 px-3 rounded-lg border border-border bg-muted/40 text-xs outline-none"
        >
          <option value="err-desc">Lỗi ↓</option>
          <option value="err-asc">Lỗi ↑</option>
          <option value="idx">Chỉ số hàng</option>
        </select>
        <button
          className="h-9 px-3 rounded-lg border border-border bg-muted/40 hover:bg-muted text-xs flex items-center gap-1.5"
          onClick={() => {
            // Chi xuat CSV tu preview/bo loc hien tai de khong tao Blob khong lo trong trinh duyet.
            const header = "row_index,reconstruction_error,prediction,prediction_label,severity";
            const rows = filtered.map((flow) =>
              `${flow.rowIndex},${flow.reconstructionError},${flow.prediction},${flow.predictionLabel},${flow.severity}`
            );
            const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = "nids_results_preview.csv";
            anchor.click();
            URL.revokeObjectURL(url);
          }}
        >
          <Download className="h-3.5 w-3.5" /> Xuất CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground bg-muted/30">
            <tr>
              <th className="text-left font-medium py-2 px-4">Hàng #</th>
              <th className="text-left font-medium py-2 px-3">Lỗi tái tạo</th>
              <th className="text-left font-medium py-2 px-3">Dự đoán</th>
              <th className="text-left font-medium py-2 px-3">Nhãn</th>
              <th className="text-left font-medium py-2 px-4">Mức độ</th>
            </tr>
          </thead>
          <tbody>
            {view.map((flow) => (
              <tr key={flow.rowIndex} className="border-t border-border hover:bg-muted/30">
                <td className="py-2 px-4 font-mono text-xs text-muted-foreground">{flow.rowIndex}</td>
                <td className={`py-2 px-3 font-mono text-xs ${flow.prediction === 1 ? "text-anomaly" : "text-muted-foreground"}`}>
                  {flow.reconstructionError.toFixed(5)}
                </td>
                <td className="py-2 px-3 font-mono text-xs">{flow.prediction}</td>
                <td className="py-2 px-3">
                  <span className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${
                    flow.prediction === 1 ? "text-anomaly bg-anomaly/10 border-anomaly/30" : "text-cyan bg-cyan/10 border-cyan/30"
                  }`}>{flow.predictionLabel === "anomaly" ? "Bất thường" : "Bình thường"}</span>
                </td>
                <td className="py-2 px-4">
                  <span className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${sevTone[flow.severity]}`}>
                    {sevLabel[flow.severity]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between p-3 px-4 border-t border-border text-xs text-muted-foreground">
        <div>
          {isPreview && (
            <p className="mb-1 text-warning">
              Đang hiển thị bản xem trước {flows.length} / {totalRows ?? flows.length} flow.
            </p>
          )}
          Hiển thị {filtered.length === 0 ? 0 : page * pageSize + 1}-{Math.min(filtered.length, (page + 1) * pageSize)} trong số {filtered.length}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setPage((current) => Math.max(0, current - 1))} className="h-7 w-7 grid place-items-center rounded-md border border-border hover:bg-muted disabled:opacity-40" disabled={page === 0}>
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span className="px-2 font-mono">{page + 1} / {pages}</span>
          <button onClick={() => setPage((current) => Math.min(pages - 1, current + 1))} className="h-7 w-7 grid place-items-center rounded-md border border-border hover:bg-muted disabled:opacity-40" disabled={page >= pages - 1}>
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
