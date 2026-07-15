import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Download, Search } from "lucide-react";
import {
  downloadUploadResultsCsv,
  type ResultsCsvPredictionFilter,
  type ResultsCsvSort,
} from "@/lib/api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { MappedFlow } from "@/lib/mapper";
import type { PaginationResponse } from "@/types/api";

interface FlowTableProps {
  uploadId: string;
  inferenceRunId: string;
  flows: MappedFlow[];
  pagination: PaginationResponse;
  loading?: boolean;
  onPageChange: (page: number) => void;
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

function formatActualLabel(flow: MappedFlow): string {
  if (flow.actualLabel !== null && flow.actualLabel !== undefined && String(flow.actualLabel).trim() !== "") {
    return String(flow.actualLabel);
  }
  if (flow.actualBinary === 0) return "Bình thường";
  if (flow.actualBinary === 1) return "Tấn công";
  return "N/A";
}

function actualLabelTone(flow: MappedFlow): string {
  if (flow.actualBinary === 0) return "text-cyan bg-cyan/10 border-cyan/30";
  if (flow.actualBinary === 1) return "text-anomaly bg-anomaly/10 border-anomaly/30";
  return "text-muted-foreground bg-muted/30 border-border";
}

export function FlowTable({
  uploadId,
  inferenceRunId,
  flows,
  pagination,
  loading = false,
  onPageChange,
}: FlowTableProps) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<"all" | "anomaly" | "normal">("all");
  const [sort, setSort] = useState<"err-desc" | "err-asc" | "idx">("err-desc");
  const [exporting, setExporting] = useState(false);

  // Các thao tác này chỉ thay đổi thứ tự/hiển thị trong trang hiện tại.
  // Chuyển trang luôn gọi backend qua onPageChange.
  const filtered = useMemo(() => {
    let nextFlows = flows;
    if (filter === "anomaly") nextFlows = nextFlows.filter((flow) => flow.prediction === 1);
    if (filter === "normal") nextFlows = nextFlows.filter((flow) => flow.prediction === 0);
    if (q) nextFlows = nextFlows.filter((flow) => String(flow.rowIndex).includes(q));
    if (sort === "err-desc") return [...nextFlows].sort((a, b) => b.reconstructionError - a.reconstructionError);
    if (sort === "err-asc") return [...nextFlows].sort((a, b) => a.reconstructionError - b.reconstructionError);
    return [...nextFlows].sort((a, b) => a.rowIndex - b.rowIndex);
  }, [q, filter, sort, flows]);

  const firstItem =
    pagination.total_items === 0 ? 0 : (pagination.page - 1) * pagination.page_size + 1;
  const lastItem = Math.min(
    pagination.total_items,
    pagination.page * pagination.page_size,
  );

  async function handleExportCsv() {
    const apiSort: ResultsCsvSort =
      sort === "err-desc" ? "err_desc" : sort === "err-asc" ? "err_asc" : "idx";

    setExporting(true);
    try {
      const { blob, filename } = await downloadUploadResultsCsv(
        uploadId,
        inferenceRunId,
        filter as ResultsCsvPredictionFilter,
        apiSort,
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card shadow-soft">
      <div className="flex flex-wrap items-center gap-3 p-4 border-b border-border">
        <div className="flex items-center gap-2 px-3 h-9 rounded-lg border border-border bg-muted/40 text-sm flex-1 min-w-[200px]">
          <Search className="h-4 w-4 text-muted-foreground" />
          <input
            value={q}
            onChange={(event) => setQ(event.target.value)}
            placeholder="Tìm theo chỉ số hàng trong trang..."
            className="bg-transparent outline-none flex-1 text-sm"
          />
        </div>
        <div className="flex rounded-lg border border-border overflow-hidden text-xs">
          {(["all", "anomaly", "normal"] as const).map((nextFilter) => (
            <button
              key={nextFilter}
              onClick={() => setFilter(nextFilter)}
              className={`px-3 h-9 capitalize transition ${filter === nextFilter ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              {nextFilter === "all" ? "Tất cả" : nextFilter === "anomaly" ? "Bất thường" : "Bình thường"}
            </button>
          ))}
        </div>
        <Select
          value={sort}
          onValueChange={(value) => setSort(value as typeof sort)}
        >
          <SelectTrigger className="h-9 w-[112px] rounded-lg border-border bg-muted/40 px-3 text-xs text-foreground shadow-none hover:bg-muted focus:ring-1 focus:ring-ring">
            <SelectValue />
          </SelectTrigger>
          <SelectContent
            align="end"
            className="z-[80] border-border bg-popover text-popover-foreground shadow-lg"
          >
            <SelectItem
              value="err-desc"
              className="text-xs text-foreground focus:bg-muted focus:text-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
            >
              Lỗi ↓
            </SelectItem>
            <SelectItem
              value="err-asc"
              className="text-xs text-foreground focus:bg-muted focus:text-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
            >
              Lỗi ↑
            </SelectItem>
            <SelectItem
              value="idx"
              className="text-xs text-foreground focus:bg-muted focus:text-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
            >
              Chỉ số hàng
            </SelectItem>
          </SelectContent>
        </Select>
        <button
          className="h-9 px-3 rounded-lg border border-border bg-muted/40 hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-xs flex items-center gap-1.5"
          disabled={exporting}
          onClick={handleExportCsv}
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
              <th className="text-left font-medium py-2 px-3">Nhãn dự đoán</th>
              <th className="text-left font-medium py-2 px-3 min-w-[150px]">Nhãn thực tế</th>
              <th className="text-left font-medium py-2 px-4">Mức độ</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((flow) => (
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
                <td className="py-2 px-3 min-w-[150px] max-w-[220px]">
                  <span
                    title={formatActualLabel(flow)}
                    className={`inline-block max-w-[200px] truncate align-middle text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${actualLabelTone(flow)}`}
                  >
                    {formatActualLabel(flow)}
                  </span>
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
          Hiển thị {firstItem}-{lastItem} trong số {pagination.total_items}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onPageChange(pagination.page - 1)}
            className="h-7 w-7 grid place-items-center rounded-md border border-border hover:bg-muted disabled:opacity-40"
            disabled={loading || !pagination.has_previous}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span className="px-2 font-mono">
            {pagination.page} / {Math.max(1, pagination.total_pages)}
          </span>
          <button
            onClick={() => onPageChange(pagination.page + 1)}
            className="h-7 w-7 grid place-items-center rounded-md border border-border hover:bg-muted disabled:opacity-40"
            disabled={loading || !pagination.has_next}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}