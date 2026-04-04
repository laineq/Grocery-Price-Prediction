import { BottomNav } from "@/components/bottom-nav";
import { BrandMark } from "@/components/branding";
import { DashboardCard } from "@/components/dashboard-card";
import { getDashboardProducts } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const dashboardCards = await getDashboardProducts();

  return (
    <main className="screen">
      <div className="phone-shell">
        <header className="page-header">
          <BrandMark />
        </header>

        <section className="hero-block">
          <h1 className="hero-title">
            <span className="hero-title__line">Predicting</span>
            <span className="hero-title__line hero-title__line--accent">
              {dashboardCards[0]?.predictionMonth ?? "No prediction yet"}
            </span>
            <span className="hero-title__line">price trends.</span>
          </h1>
          <p className="hero-subtitle">
            AI-powered forecasting of Canadian grocery prices using real-world data.
          </p>
        </section>

        <section className="market-grid">
          {dashboardCards.map((product) => (
            <DashboardCard key={product.key} product={product} />
          ))}
        </section>

        <BottomNav />
      </div>
    </main>
  );
}
