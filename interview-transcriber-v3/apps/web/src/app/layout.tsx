import type { Metadata } from "next";

import { Providers } from "@/providers";
import { cn } from "@/lib/utils";

import "./globals.css";

export const metadata: Metadata = {
  title: "Transcriber — транскрибация интервью",
  description:
    "Автоматическая транскрибация длинных интервью и разговоров. Русский язык, высокая точность.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className={cn("min-h-screen font-sans antialiased")}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
