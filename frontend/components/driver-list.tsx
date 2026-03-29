import { AnalyticsIcon, BoxIcon, ExchangeIcon, GasIcon, WeatherIcon } from "@/components/icons";
import type { ProductSummary } from "@/lib/data";

const iconMap = {
  weather: WeatherIcon,
  exchange: ExchangeIcon,
  gas: GasIcon,
  import: BoxIcon,
  cpi: AnalyticsIcon
};

export function DriverList({ product }: { product: ProductSummary }) {
  return (
    <section className="detail-section detail-section--soft">
      <h2 className="detail-section__title">Drivers of Change</h2>

      <div className="drivers">
        {product.drivers.map((driver) => {
          const Icon = iconMap[driver.icon];

          return (
            <div key={driver.title} className="driver-item">
              <div className="driver-item__icon-wrap">
                <Icon
                  className="driver-item__icon"
                  stroke={product.accent === "green" ? "#567f3e" : "#ca2d26"}
                />
              </div>

              <div>
                <div className="driver-item__title">{driver.title}</div>
                <div className="driver-item__description">{driver.description}</div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
