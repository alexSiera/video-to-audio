export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 p-8">
      <h1 className="text-4xl font-bold tracking-tight">Transcriber</h1>
      <p className="text-muted-foreground text-center max-w-md">
        Автоматическая транскрибация длинных интервью и разговоров. Сервис запускается — скоро здесь
        будет загрузка и редактор.
      </p>
    </main>
  );
}
