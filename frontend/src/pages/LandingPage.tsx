import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Brain,
  Cpu,
  LineChart,
  Lock,
  Network,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";

const CHART_WIDTH = 320;
const CHART_HEIGHT = 132;
const CHART_PADDING = 14;
const CHART_BOTTOM = CHART_HEIGHT - CHART_PADDING;
const CHART_THRESHOLD = 6.0313;
const CHART_MAX_ERROR = 8.4;

// Du lieu demo gom 50 mau luong, giu cac diem dot bien vuot nguong de minh hoa bat thuong.
const demoTimelineScores = [
  1.2, 1.5, 1.4, 1.7, 1.9, 1.6, 2.0, 2.2, 1.8, 2.1,
  2.4, 2.0, 2.3, 2.7, 2.5, 2.2, 2.6, 2.8, 3.0, 2.7,
  3.1, 3.4, 2.9, 3.2, 3.6, 3.1, 3.4, 7.2, 5.9, 3.3,
  3.0, 3.5, 3.8, 3.2, 3.6, 4.0, 3.7, 7.8, 6.6, 4.1,
  3.8, 4.2, 3.9, 4.4, 4.1, 6.9, 7.5, 4.3, 3.9, 4.2,
];

type TimelinePoint = {
  x: number;
  y: number;
  score: number;
};

// Chuyen diem loi tai tao sang toa do SVG, dao truc Y de diem cao nam o phia tren bieu do.
const timelinePoints: TimelinePoint[] = demoTimelineScores.map((score, index) => {
  const x = CHART_PADDING + (index / (demoTimelineScores.length - 1)) * (CHART_WIDTH - CHART_PADDING * 2);
  const y = CHART_BOTTOM - (score / CHART_MAX_ERROR) * (CHART_HEIGHT - CHART_PADDING * 2);
  return { x, y, score };
});

// Tao path muot bang cubic Bezier de timeline trong gon hon so voi cot roi rac.
function createSmoothPath(points: TimelinePoint[]) {
  return points.reduce((path, point, index) => {
    if (index === 0) {
      return `M ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
    }

    const previousPoint = points[index - 1];
    const controlDistance = (point.x - previousPoint.x) / 2;
    return `${path} C ${(previousPoint.x + controlDistance).toFixed(1)} ${previousPoint.y.toFixed(1)}, ${(point.x - controlDistance).toFixed(1)} ${point.y.toFixed(1)}, ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
  }, "");
}

const timelinePath = createSmoothPath(timelinePoints);
const timelineAreaPath = `${timelinePath} L ${timelinePoints[timelinePoints.length - 1].x.toFixed(1)} ${CHART_BOTTOM} L ${timelinePoints[0].x.toFixed(1)} ${CHART_BOTTOM} Z`;
const thresholdY = CHART_BOTTOM - (CHART_THRESHOLD / CHART_MAX_ERROR) * (CHART_HEIGHT - CHART_PADDING * 2);

// Tach cac doan co diem bat thuong de ve lop mau do phu len duong cyan.
const anomalySegments = timelinePoints
  .map((point, index) => ({ point, index }))
  .filter(({ point }) => point.score > CHART_THRESHOLD)
  .map(({ index }) => timelinePoints.slice(Math.max(0, index - 1), Math.min(timelinePoints.length, index + 2)))
  .filter((segment) => segment.length > 1);

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      <style>{`
        @keyframes draw-anomaly-timeline {
          to {
            stroke-dashoffset: 0;
          }
        }

        .landing-timeline-path {
          stroke-dasharray: 720;
          stroke-dashoffset: 720;
          animation: draw-anomaly-timeline 1.2s ease-out forwards;
        }
      `}</style>
      <div className="pointer-events-none absolute inset-0 bg-glow-field" />
      <div className="pointer-events-none absolute inset-0 grid-bg opacity-40" />

      {/* Nav */}
      <header className="relative z-10">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-gradient-primary grid place-items-center shadow-glow">
              <ShieldCheck className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-sm">VAE NIDS</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Phát hiện bất thường mạng</div>
            </div>
          </Link>
          <nav className="hidden md:flex items-center gap-7 text-sm text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition">Tính năng</a>
            <a href="#architecture" className="hover:text-foreground transition">Kiến trúc</a>
            <a href="#metrics" className="hover:text-foreground transition">Chỉ số</a>
          </nav>
          <Link to="/dashboard" className="text-sm flex items-center gap-1.5 px-4 h-9 rounded-lg bg-card border border-border hover:border-primary/50 transition">
            Mở bảng điều khiển <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 pt-16 pb-24">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-card/70 text-xs">
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success pulse-dot text-success" />
              <span className="text-muted-foreground">CICIDS2017 · VAE đã huấn luyện · sẵn sàng phân tích</span>
            </div>
            <h1 className="mt-6 text-4xl md:text-5xl xl:text-6xl font-semibold tracking-tight leading-[1.05] max-w-2xl">
              Phát hiện bất thường lưu lượng mạng<br />
              <span className="text-gradient">bằng Variational Autoencoder (VAE)</span>
            </h1>
            <p className="mt-5 text-base md:text-lg text-muted-foreground max-w-xl leading-relaxed">
              Phát hiện hành vi mạng bất thường bằng học sâu không giám sát. Mô hình autoencoder biến phân
              đã huấn luyện chấm điểm từng luồng so với baseline lưu lượng bình thường đã học.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/upload" className="group flex items-center gap-2 px-5 h-11 rounded-lg bg-gradient-primary text-primary-foreground font-medium text-sm shadow-glow hover:opacity-95">
                Bắt đầu phân tích <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition" />
              </Link>
              <Link to="/dashboard" className="flex items-center gap-2 px-5 h-11 rounded-lg border border-border bg-card text-sm hover:border-primary/50 transition">
                Xem dashboard
              </Link>
            </div>

            {/* Các chỉ số đã xác minh từ artifacts/threshold/evaluation_metrics.json và artifacts/models/model_config.json */}
            <div className="mt-10 grid grid-cols-3 gap-6 max-w-md">
              {[
                { v: "64.3%", l: "F1 Score" },
                { v: "77.7%", l: "ROC AUC" },
                { v: "59.2%", l: "Recall" },
              ].map((s) => (
                <div key={s.l}>
                  <div className="text-2xl font-semibold font-mono">{s.v}</div>
                  <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">{s.l}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right visual */}
          <div className="relative">
            <div className="relative rounded-2xl border border-border bg-card/80 backdrop-blur p-5 shadow-glow">
              <div className="flex items-center justify-between mb-4">
                {/* Panel demo minh hoạ lỗi tái tạo theo từng luồng CSV */}
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-primary" />
                  <span className="text-xs font-medium">Phân tích lỗi tái tạo VAE</span>
                </div>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                  demo kết quả CSV
                </span>
              </div>
              {/* Chu giai nho giup nguoi xem phan biet luong binh thuong, bat thuong va nguong phat hien. */}
              <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-muted-foreground">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-cyan" />
                  <span>Bình thường</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-anomaly" />
                  <span>Bất thường</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="h-px w-6 border-t border-dashed border-muted-foreground/70" />
                  <span>Ngưỡng phát hiện</span>
                </div>
              </div>

              <div className="relative h-40 overflow-hidden rounded-xl border border-border/70 bg-background/35">
                {/* SVG timeline: dien tich nen mem, duong cyan chinh va lop do cho cac spike vuot nguong. */}
                <svg className="h-full w-full" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="Timeline điểm lỗi tái tạo VAE">
                  <defs>
                    <linearGradient id="timelineAreaGradient" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stopColor="var(--cyan)" stopOpacity="0.28" />
                      <stop offset="100%" stopColor="var(--cyan)" stopOpacity="0.02" />
                    </linearGradient>
                    <filter id="timelineGlow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="2.5" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>

                  <path d={timelineAreaPath} fill="url(#timelineAreaGradient)" />
                  <line x1={CHART_PADDING} x2={CHART_WIDTH - CHART_PADDING} y1={thresholdY} y2={thresholdY} stroke="var(--muted-foreground)" strokeWidth="1" strokeDasharray="5 5" opacity="0.7" />
                  <text x={CHART_WIDTH - CHART_PADDING - 42} y={thresholdY - 6} fill="var(--muted-foreground)" fontSize="10" fontFamily="JetBrains Mono, monospace">
                    
                  </text>
                  <path d={timelinePath} className="landing-timeline-path" fill="none" stroke="var(--cyan)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" filter="url(#timelineGlow)" />
                  {anomalySegments.map((segment, index) => (
                    <path
                      key={`${segment[0].x}-${index}`}
                      d={createSmoothPath(segment)}
                      className="landing-timeline-path"
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  ))}
                  {timelinePoints
                    .filter((point) => point.score > CHART_THRESHOLD)
                    .map((point) => (
                      <circle key={`${point.x}-${point.y}`} cx={point.x} cy={point.y} r="3.2" fill="#ef4444" stroke="var(--card)" strokeWidth="1.5" />
                    ))}
                </svg>
              </div>

              {/* Số liệu từ fixed_batch.csv (128 luồng, 13 bất thường) và threshold.json */}
              <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
                <div className="rounded-lg border border-border border-l-2 border-l-cyan bg-background/25 p-3">
                  <div className="text-muted-foreground text-[10px] uppercase tracking-wider">Tổng luồng</div>
                  <div className="font-mono text-lg">128</div>
                </div>
                <div className="rounded-lg border border-border border-l-2 border-l-anomaly bg-background/25 p-3">
                  <div className="text-muted-foreground text-[10px] uppercase tracking-wider">Bất thường</div>
                  <div className="font-mono text-lg text-anomaly">13</div>
                </div>
                <div className="rounded-lg border border-border border-l-2 border-l-muted-foreground/60 bg-background/25 p-3">
                  <div className="text-muted-foreground text-[10px] uppercase tracking-wider">Ngưỡng (p99)</div>
                  <div className="font-mono text-lg">6.0313</div>
                </div>
              </div>
            </div>

            {/* floating cards */}
            {/* Card demo minh hoạ: luồng được phát hiện bất thường (không phải dữ liệu thực) */}
            <div className="hidden md:block absolute -bottom-6 -left-6 rounded-xl border border-border bg-card p-3 shadow-soft animate-float">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-anomaly/15 text-anomaly grid place-items-center">
                  <Zap className="h-4 w-4" />
                </div>
                <div className="text-xs">
                  <div className="font-medium">Luồng bất thường</div>
                  <div className="text-muted-foreground">Lỗi tái tạo &gt; ngưỡng</div>
                </div>
              </div>
            </div>
            <div className="hidden md:block absolute -top-4 -right-4 rounded-xl border border-border bg-card p-3 shadow-soft">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <div className="text-xs">
                  <div className="font-medium">Không gian latent 16d</div>
                  <div className="text-muted-foreground">66 đặc trưng → mã hoá VAE</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-6 py-20 border-t border-border">
        <div className="max-w-2xl">
          <div className="text-xs uppercase tracking-[0.16em] text-primary">Khả năng</div>
          <h2 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">Được xây dựng cho phân tích lưu lượng mạng</h2>
          <p className="mt-3 text-muted-foreground">Tải lên CSV lưu lượng mạng và nhận kết quả phân tích bất thường dựa trên mô hình VAE đã huấn luyện trên tập CICIDS2017.</p>
        </div>

        <div className="mt-10 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { i: Network, t: "Baseline không giám sát", b: "VAE học phân bố lưu lượng bình thường — không cần nhãn tấn công." },
            { i: Brain, t: "Chấm điểm tái tạo", b: "Lỗi theo từng luồng so với không gian latent đã học, So sánh với ngưỡng đã xác định từ tập validation.." },
            // Phân loại batch: hệ thống hoạt động theo lô CSV, không phải thời gian thực
            { i: Activity, t: "Phân loại theo lô (Batch)", b: "Mỗi luồng trong file CSV được chấm điểm riêng biệt và phân loại là bình thường hoặc bất thường." },
            { i: LineChart, t: "Biểu đồ phân bố", b: "Histogram lỗi tái tạo và bảng kết quả theo từng luồng, hiển thị trực tiếp trên dashboard." },
            { i: Lock, t: "Artifact có thể tái tạo", b: "Backend FastAPI + PyTorch, scaler và ngưỡng được lưu thành artifact để đảm bảo nhất quán giữa training và inference." },
            { i: Cpu, t: "Pipeline nhất quán", b: "Cùng bước tiền xử lý (imputation → StandardScaler) được dùng trong cả training lẫn inference." },
          ].map((f) => {
            const Icon = f.i;
            return (
              <div key={f.t} className="group relative rounded-xl border border-border bg-card p-5 hover:border-primary/40 hover:shadow-glow transition">
                <div className="h-10 w-10 rounded-lg bg-gradient-primary/10 border border-primary/20 grid place-items-center text-primary">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-base font-semibold tracking-tight">{f.t}</h3>
                <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">{f.b}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Architecture */}
      <section id="architecture" className="relative z-10 max-w-7xl mx-auto px-6 py-20 border-t border-border">
        <div className="grid lg:grid-cols-2 gap-12">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-primary">Kiến trúc</div>
            <h2 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">VAE đã huấn luyện ở trung tâm</h2>
            {/* Mô tả kiến trúc dựa trên artifacts/models/model_config.json: input_dim=66, latent_dim=16, hidden_dims=[128,64] */}
            <p className="mt-3 text-muted-foreground leading-relaxed">
              Luồng mạng được chuẩn hóa bằng scaler đã khớp, mã hóa qua hai lớp ẩn vào không gian latent 16 chiều, rồi
              tái tạo. Luồng mà bộ giải mã không thể tái tạo trung thực sẽ có lỗi tái tạo vượt ngưỡng
              và được phân loại là bất thường.
            </p>
            <ul className="mt-6 space-y-3 text-sm">
              {["Nhập CSV & kiểm tra schema", "Chuẩn hóa đặc trưng (StandardScaler)", "Bộ mã hóa VAE → latent μ, σ (16d)", "Bộ giải mã → lỗi tái tạo MSE", "So sánh với ngưỡng p99 = 6.0313"].map((s, i) => (
                <li key={s} className="flex items-center gap-3">
                  <div className="h-6 w-6 rounded-full bg-card border border-primary/40 text-primary text-[11px] font-mono grid place-items-center">{i + 1}</div>
                  <span>{s}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-2xl border border-border bg-card p-6 shadow-soft">
            <div className="space-y-3 text-xs font-mono">
              {/* Kiến trúc chính xác: model_config.json → input_dim=66, latent_dim=16, hidden=[128,64]; threshold.json → p95=3.218 */}
              {[
                { l: "Đầu vào", v: "CSV flow → 66 đặc trưng" },
                { l: "Mã hóa", v: "66 → 128 → 64 → (μ=16, σ=16)" },
                { l: "Latent", v: "z = μ (inference, tất định)" },
                { l: "Giải mã", v: "16 → 64 → 128 → 66 tái tạo" },
                { l: "Điểm", v: "MSE(x, x̂) mỗi luồng" },
                { l: "Ngưỡng", v: "p99 lỗi validation ≈ 6.0313" },
              ].map((s, i) => (
                <div key={s.l} className="relative">
                  <div className="flex items-center gap-3 rounded-lg border border-border bg-background/40 px-4 py-3">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground w-20">{s.l}</div>
                    <div className="flex-1 text-sm">{s.v}</div>
                  </div>
                  {i < 5 && <div className="absolute left-6 -bottom-3 h-3 w-px bg-border" />}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section id="metrics" className="relative z-10 max-w-7xl mx-auto px-6 py-20 border-t border-border">
        <div className="rounded-2xl p-px bg-gradient-primary shadow-glow">
          <div className="rounded-[15px] bg-card p-10 md:p-14 text-center">
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">Sẵn sàng kiểm tra lưu lượng của bạn?</h2>
            <p className="mt-3 text-muted-foreground max-w-xl mx-auto">
              Tải lên CSV luồng mạng. VAE sẽ trả về lỗi tái tạo, kết quả phát hiện bất thường và thông tin AI trong vài giây.
            </p>
            <div className="mt-7 flex flex-wrap justify-center gap-3">
              <Link to="/upload" className="flex items-center gap-2 px-6 h-11 rounded-lg bg-gradient-primary text-primary-foreground font-medium text-sm shadow-glow">
                Bắt đầu phân tích <ArrowRight className="h-4 w-4" />
              </Link>
              <Link to="/dashboard" className="flex items-center gap-2 px-6 h-11 rounded-lg border border-border bg-background/40 text-sm">
                Xem dashboard
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="relative z-10 border-t border-border">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
          <div>© 2026 VAE NIDS · Hệ thống phát hiện bất thường lưu lượng mạng</div>
          <div className="font-mono">VAE-NIDS · FastAPI · PyTorch · CICIDS2017</div>
        </div>
      </footer>
    </div>
  );
}
