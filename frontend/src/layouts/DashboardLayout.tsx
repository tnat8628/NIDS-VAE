import type { ReactNode } from 'react'
import AppSidebar from '@/components/common/app-sidebar'

interface DashboardLayoutProps {
  children: ReactNode
}

/**
 * Shell layout dùng chung cho tất cả các trang dashboard.
 * Bao gồm AppSidebar bên trái và vùng nội dung chính bên phải.
 * Trang landing page (/) KHÔNG sử dụng layout này.
 */
export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    // Bọc toàn màn hình, ngăn tràn ngang
    <div className="flex min-h-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar cố định bên trái – ẩn trên mobile (hidden md:flex bên trong AppSidebar) */}
      <AppSidebar />

      {/* Vùng nội dung chính – chiếm không gian còn lại, cho phép cuộn dọc */}
      <div className="flex flex-1 flex-col min-w-0 overflow-y-auto">
        {children}
      </div>
    </div>
  )
}
