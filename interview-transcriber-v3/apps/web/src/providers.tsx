"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";

/**
 * Корневой клиентский Provider. Оборачивает всё приложение:
 * - TanStack Query (server-state)
 *
 * Zustand-сторы инициализируются лениво при импорте; здесь их оборачивать не нужно.
 * Theme provider — добавляется при интеграции next-themes.
 */
export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 мин
            retry: 1,
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
