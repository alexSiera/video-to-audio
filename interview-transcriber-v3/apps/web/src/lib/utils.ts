import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Объединяет Tailwind-классы с интеллектуальным слиянием конфликтующих классов.
 * Используется повсеместно вместо шаблонного `clsx + twMerge`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
