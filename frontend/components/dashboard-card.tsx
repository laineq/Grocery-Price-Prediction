import Link from "next/link";
import { ArrowRightIcon } from "@/components/icons";
import type { ProductSummary } from "@/lib/data";

export function DashboardCard({ product }: { product: ProductSummary }) {
  const positive = product.changePct >= 0;

  return (
    <article className="market-card">
      <div className="market-card__hero" style={{ background: product.heroGradient }}>
        <div className="market-card__emoji">{product.imageEmoji}</div>
      </div>

      <div className="market-card__body">
        <div>
          <h2 className="market-card__title">{product.name}</h2>
          <p className="market-card__meta">{product.unitLabel}</p>
        </div>

        <div className="market-card__price-block">
          <div className="market-card__label">Forecast Value</div>
          <div
            className={`market-card__price ${product.accent === "green" ? "is-green" : "is-red"}`}
          >
            C${product.forecastPrice.toFixed(2)}
          </div>
          <div className={`market-card__delta ${positive ? "is-up" : "is-down"}`}>
            {positive ? "+" : "-"}
            {Math.abs(product.changePct).toFixed(1)}%
          </div>
        </div>
      </div>

      <Link
        href={`/${product.key}`}
        className={`market-card__button ${product.accent === "green" ? "is-green" : "is-red"}`}
      >
        View Details
        <ArrowRightIcon className="market-card__button-icon" />
      </Link>
    </article>
  );
}
