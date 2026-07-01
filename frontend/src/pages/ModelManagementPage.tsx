import { Topbar } from "@/components/common/topbar";
import { Cpu, GitBranch, Sliders, Package } from "lucide-react";

const versions = [
  { v: "vae-nids-v1.4.2", status: "active", auc: 0.964, trained: "2026-05-14", thr: 0.0428 },
  { v: "vae-nids-v1.4.1", status: "archived", auc: 0.958, trained: "2026-04-02", thr: 0.0461 },
  { v: "vae-nids-v1.3.7", status: "archived", auc: 0.941, trained: "2026-02-21", thr: 0.0512 },
];

export default function ModelManagementPage() {
  return (
    <>
      <Topbar title="Quản lý mô hình" subtitle="Đăng ký phiên bản, huấn luyện lại và điều chỉnh ngưỡng" />
      <main className="flex-1 px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6 max-w-[1280px] mx-auto w-full">
        <div className="grid lg:grid-cols-3 gap-4">
          {[
            { i: Cpu, t: "Huấn luyện lại", b: "Lên lịch huấn luyện không giám sát trên cửa sổ baseline mới nhất." },
            { i: Sliders, t: "Điều chỉnh ngưỡng", b: "Điều chỉnh ngưỡng hoạt động hoặc chuyển sang chế độ phân vị/percentile." },
            { i: Package, t: "Kho artifact", b: "Thăng cấp, lưu trữ và khôi phục artifact mô hình." },
          ].map((c) => {
            const Icon = c.i;
            return (
              <div key={c.t} className="relative rounded-xl border border-border bg-card p-5 overflow-hidden">
                <div className="absolute top-3 right-3 text-[9px] uppercase tracking-[0.14em] px-1.5 py-0.5 rounded-full bg-violet/15 text-violet border border-violet/30">Sắp có</div>
                <Icon className="h-5 w-5 text-primary" />
                <h3 className="mt-3 text-sm font-semibold">{c.t}</h3>
                <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{c.b}</p>
              </div>
            );
          })}
        </div>

        <div className="rounded-xl border border-border bg-card shadow-soft overflow-hidden">
          <div className="p-5 border-b border-border flex items-center gap-2">
            <GitBranch className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">Đăng ký phiên bản</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground bg-muted/30">
                <tr>
                  <th className="text-left font-medium py-2 px-5">Phiên bản</th>
                  <th className="text-left font-medium py-2 px-3">Trạng thái</th>
                  <th className="text-left font-medium py-2 px-3">ROC-AUC</th>
                  <th className="text-left font-medium py-2 px-3">Ngưỡng</th>
                  <th className="text-left font-medium py-2 px-5">Ngày huấn luyện</th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.v} className="border-t border-border hover:bg-muted/30">
                    <td className="py-2.5 px-5 font-mono text-xs">{v.v}</td>
                    <td className="py-2.5 px-3">
                      <span className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full border ${
                        v.status === "active" ? "text-success bg-success/10 border-success/30" : "text-muted-foreground bg-muted/40 border-border"
                      }`}>{v.status === "active" ? "Hoạt động" : "Lưu trữ"}</span>
                    </td>
                    <td className="py-2.5 px-3 font-mono text-xs">{v.auc.toFixed(3)}</td>
                    <td className="py-2.5 px-3 font-mono text-xs">{v.thr.toFixed(4)}</td>
                    <td className="py-2.5 px-5 text-xs text-muted-foreground">{v.trained}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </>
  );
}
