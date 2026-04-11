import Link from "next/link";
import { BottomNav } from "@/components/bottom-nav";
import { BrandMark } from "@/components/branding";
import { DetailChart } from "@/components/detail-chart";
import { DriverList } from "@/components/driver-list";
import { ArrowLeftIcon } from "@/components/icons";
import type { ProductSummary } from "@/lib/data";

export function DetailPage({ product }: { product: ProductSummary }) {
  const positive = product.changePct >= 0;

  return (
    <main className="screen">
      <div className="phone-shell">
        <header className="page-header">
          <BrandMark />
        </header>

        <Link href="/" className="back-link">
          <ArrowLeftIcon className="back-link__icon" />
          <span>Back to Market Dashboard</span>
        </Link>

        <section className="hero-block hero-block--detail">
          <div className={`detail-tag ${product.accent === "green" ? "is-green" : "is-red"}`}>
            {product.detailTag}
          </div>
          <h1 className="hero-title hero-title--detail" style={{ marginTop: 18 }}>
            {product.name} Prediction
          </h1>
          <p className="hero-subtitle" style={{ marginTop: 14 }}>
            Data-driven analysis of Canadian price trends to forecast next month&apos;s
            volatility.
          </p>
        </section>

        <section className="projected-card">
          <div className="projected-card__eyebrow">Projected Price</div>
          <div className={`projected-card__badge ${product.accent === "green" ? "is-green" : "is-red"}`}>
            {product.predictionMonth.toUpperCase()} Prediction
          </div>

          <div className="projected-card__value-row">
            <div
              className={`projected-card__price ${product.accent === "green" ? "is-green" : "is-red"}`}
            >
              C${product.forecastPrice.toFixed(2)}
            </div>
            <div className={`projected-card__delta ${positive ? "is-up" : "is-down"}`}>
              {positive ? "+" : ""}
              {product.changePct.toFixed(1)}%
            </div>
          </div>
        </section>

        <DetailChart product={product} />

        <DriverList product={product} />

        <BottomNav />
      </div>
    </main>
  );
}
