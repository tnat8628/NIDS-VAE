import { Topbar } from "@/components/common/topbar";
import { HealthStatusCards } from "@/components/dashboard/health-status-card";
import { health, stats } from "@/data/mockData";
import { Cpu, Database, GitBranch, Sigma } from "lucide-react";

export default function HealthPage() {
  const meta = [
    { i: GitBranch, l: "Phiên bản mô hình", v: stats.modelVersion },
    { i: Sigma, l: "Ngưỡng hoạt động", v: stats.threshold.toFixed(4) },
    { i: Cpu, l: "Độ trễ suy luận", v: `${stats.latencyMs}ms p50` },
    { i: Database, l: "Kho artifact", v: "s3://nids-artifacts/v1.4.2" },
  ];
  return (
    <>
      <Topbar title="Trạng thái hệ thống" subtitle="Trạng thái dịch vụ, artifact mô hình và quan sát hệ thống" />
      <main className="flex-1 px-4 md:px-6 lg:px-8 py-4 md:py-6 space-y-4 lg:space-y-6 max-w-[1280px] mx-auto w-full">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {meta.map((m) => {
            const Icon = m.i;
            return (
              <div key={m.l} className="rounded-xl border border-border bg-card p-5 overflow-hidden">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Icon className="h-4 w-4" />
                  <span className="text-[11px] uppercase tracking-[0.14em]">{m.l}</span>
                </div>
                <div className="mt-2 font-mono text-sm break-all">{m.v}</div>
              </div>
            );
          })}
        </div>

        <div>
          <h3 className="text-sm font-semibold mb-3 px-1">Dịch vụ</h3>
          <HealthStatusCards items={health} />
        </div>
      </main>
    </>
  );
}
