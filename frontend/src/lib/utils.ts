import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Kết hợp class Tailwind CSS an toàn – tránh xung đột class.
 * Dùng thay cho clsx thuần túy ở mọi nơi trong dự án.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}
