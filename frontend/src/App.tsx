import { Routes, Route } from 'react-router-dom'
import DashboardLayout from '@/layouts/DashboardLayout'
import LandingPage from '@/pages/LandingPage'
import DashboardPage from '@/pages/DashboardPage'
import UploadPage from '@/pages/UploadPage'
import UploadsPage from '@/pages/UploadsPage'
import GlobalFlowsPage from '@/pages/GlobalFlowsPage'
import ResultsPage from '@/pages/ResultsPage'
import HealthPage from '@/pages/HealthPage'
import ModelManagementPage from '@/pages/ModelManagementPage'
import { Toaster } from '@/components/ui/sonner'

/**
 * Định nghĩa tất cả route của ứng dụng NIDS VAE Dashboard.
 * - Trang `/` render LandingPage độc lập (KHÔNG có sidebar).
 * - Tất cả route còn lại được bọc trong DashboardLayout (có AppSidebar).
 * Sử dụng React Router v6 (KHÔNG dùng TanStack Router).
 */
function App() {
  return (
    <>
      <Routes>
        {/* Trang chủ – standalone, không sidebar */}
        <Route path="/" element={<LandingPage />} />

        {/* Các trang ứng dụng – bọc trong DashboardLayout để hiển thị sidebar */}
        <Route path="/dashboard" element={<DashboardLayout><DashboardPage /></DashboardLayout>} />
        <Route path="/upload" element={<DashboardLayout><UploadPage /></DashboardLayout>} />
        <Route path="/uploads" element={<DashboardLayout><UploadsPage /></DashboardLayout>} />
        <Route path="/flows" element={<DashboardLayout><GlobalFlowsPage /></DashboardLayout>} />
        <Route path="/results" element={<DashboardLayout><ResultsPage /></DashboardLayout>} />
        <Route path="/health" element={<DashboardLayout><HealthPage /></DashboardLayout>} />
        {/* <Route path="/analytics" element={<DashboardLayout><AnalyticsPage /></DashboardLayout>} /> */}
        <Route path="/models" element={<DashboardLayout><ModelManagementPage /></DashboardLayout>} />
      </Routes>
      <Toaster richColors />
    </>
  )
}

export default App