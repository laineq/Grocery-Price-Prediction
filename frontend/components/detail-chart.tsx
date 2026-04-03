"use client";

import { useEffect, useMemo, useState } from "react";
import type { ProductSummary } from "@/lib/data";

type RangeKey = "1Y" | "5Y" | "ALL";

type ChartPoint = {
  date: string;
  price: number;
  forecast?: boolean;
  lowerBound?: number;
  upperBound?: number;
};

type ChartPointWithCoords = ChartPoint & {
  x: number;
  y: number;
  phase: "historical" | "delayed" | "forecast";
};

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function formatDate(date: string) {
  const [year, month] = date.split("-").map(Number);
  return `${MONTH_NAMES[month - 1]} ${year}`;
}

function rangeCount(range: RangeKey) {
  if (range === "1Y") return 12;
  if (range === "5Y") return 60;
  return Number.POSITIVE_INFINITY;
}

function getYBounds(points: ChartPoint[]) {
  const values = points.map((point) => point.price);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * 0.15, 0.12);
  return { min: min - pad, max: max + pad };
}

function getCurrentMonthKey() {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = `${now.getUTCMonth() + 1}`.padStart(2, "0");
  return `${year}-${month}`;
}

function buildPath(points: ChartPointWithCoords[]) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");
}

export function DetailChart({ product }: { product: ProductSummary }) {
  const [range, setRange] = useState<RangeKey>("1Y");
  const currentMonthKey = getCurrentMonthKey();

  const points = useMemo(() => {
    const count = rangeCount(range);
    if (!Number.isFinite(count) || product.chart.length <= count) {
      return product.chart;
    }
    return product.chart.slice(-count);
  }, [product.chart, range]);

  const [activeDate, setActiveDate] = useState(points[points.length - 1]?.date ?? "");

  useEffect(() => {
    setActiveDate(points[points.length - 1]?.date ?? "");
  }, [points]);

  const active = useMemo(() => {
    return points.find((point) => point.date === activeDate) ?? points[points.length - 1];
  }, [activeDate, points]);

  const geometry = useMemo(() => {
    const width = 320;
    const height = 360;
    const padding = { top: 26, right: 14, bottom: 34, left: 12 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const bounds = getYBounds(points);

    const mapped = points.map((point, index) => {
      const x = padding.left + (points.length === 1 ? innerWidth / 2 : (innerWidth * index) / (points.length - 1));
      const normalized = (point.price - bounds.min) / (bounds.max - bounds.min || 1);
      const y = padding.top + innerHeight - normalized * innerHeight;
      const phase: ChartPointWithCoords["phase"] = !point.forecast
        ? "historical"
        : point.date <= currentMonthKey
          ? "delayed"
          : "forecast";
      return { ...point, x, y, phase };
    });

    const historical = mapped.filter((point) => point.phase === "historical");
    const delayed = mapped.filter((point) => point.phase === "delayed");
    const forecast = mapped.filter((point) => point.phase === "forecast");
    const activePoint = mapped.find((point) => point.date === active.date) ?? mapped[mapped.length - 1];

    const delayedPathPoints =
      delayed.length > 0
        ? [historical[historical.length - 1], ...delayed].filter(Boolean) as ChartPointWithCoords[]
        : [];
    const forecastPathPoints =
      forecast.length > 0
        ? [
            delayed[delayed.length - 1] ?? historical[historical.length - 1],
            ...forecast,
          ].filter(Boolean) as ChartPointWithCoords[]
        : [];

    return {
      width,
      height,
      padding,
      mapped,
      historical,
      delayed,
      forecast,
      activePoint,
      delayedPathPoints,
      forecastPathPoints
    };
  }, [active.date, currentMonthKey, points]);

  const historicalPath = buildPath(geometry.historical);
  const delayedPath = buildPath(geometry.delayedPathPoints);
  const forecastPath = buildPath(geometry.forecastPathPoints);

  const activeLeft = (geometry.activePoint.x / geometry.width) * 100;
  const tooltipClassName = [
    "chart-tooltip",
    geometry.activePoint.phase === "delayed"
      ? "is-delayed"
      : product.accent === "green"
        ? "is-green"
        : "is-red",
    activeLeft > 84 ? "is-right-edge" : "",
    activeLeft < 16 ? "is-left-edge" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="detail-section chart-card">
      <div className="detail-section__header">
        <div>
          <h2 className="detail-section__title">Price Movement Index</h2>
          <div className="detail-section__unit">
            C$ / {product.key === "avocado" ? "UNIT" : "KG"}
          </div>
        </div>

        <div className="detail-section__range">
          {(["1Y", "5Y", "ALL"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={`pill ${range === item ? `is-active is-${product.accent}` : ""}`}
              onClick={() => setRange(item)}
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-plot">
        <div
          className={tooltipClassName}
          style={{
            left: `${activeLeft}%`,
            top: `${(geometry.activePoint.y / geometry.height) * 100}%`
          }}
        >
          C${geometry.activePoint.price.toFixed(2)} ({formatDate(geometry.activePoint.date).toUpperCase()})
        </div>

        <svg
          className="chart-svg"
          viewBox={`0 0 ${geometry.width} ${geometry.height}`}
          role="img"
          aria-label={`${product.name} price movement chart`}
        >
          {[0, 1, 2, 3, 4].map((line) => {
            const y = geometry.padding.top + ((geometry.height - geometry.padding.top - geometry.padding.bottom) / 4) * line;
            return (
              <line
                key={line}
                x1={geometry.padding.left}
                x2={geometry.width - geometry.padding.right}
                y1={y}
                y2={y}
                stroke="rgba(31, 37, 34, 0.08)"
                strokeWidth="1"
              />
            );
          })}

          <path
            d={historicalPath}
            fill="none"
            stroke={product.accent === "green" ? "#567f3e" : "#ca2d26"}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {delayedPath ? (
            <path
              d={delayedPath}
              fill="none"
              stroke="#b58b31"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : null}

          {forecastPath ? (
            <path
              d={forecastPath}
              fill="none"
              stroke={product.accent === "green" ? "#567f3e" : "#ca2d26"}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="7 7"
            />
          ) : null}

          {geometry.mapped.map((point) => {
            const isActive = point.date === active.date;
            const isForecast = point.phase !== "historical";
            const stroke =
              point.phase === "delayed"
                ? "#b58b31"
                : product.accent === "green"
                  ? "#567f3e"
                  : "#ca2d26";
            const activeFill =
              point.phase === "delayed"
                ? "rgba(181, 139, 49, 0.14)"
                : product.accent === "green"
                  ? "rgba(86, 127, 62, 0.12)"
                  : "rgba(202, 45, 38, 0.12)";

            return (
              <g
                key={point.date}
                onClick={() => setActiveDate(point.date)}
                style={{ cursor: "pointer" }}
              >
                {isActive ? (
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r="11"
                    fill={activeFill}
                  />
                ) : null}
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={isActive ? 7 : 5}
                  fill="#fff"
                  stroke={stroke}
                  strokeWidth={isForecast ? 4 : 3}
                />
              </g>
            );
          })}

          {geometry.forecast[0] ? (
            <rect
              x={(geometry.delayed[geometry.delayed.length - 1] ?? geometry.historical[geometry.historical.length - 1]).x}
              y={geometry.forecast[0].y}
              width={
                geometry.forecast[0].x -
                (geometry.delayed[geometry.delayed.length - 1] ?? geometry.historical[geometry.historical.length - 1]).x
              }
              height={geometry.height - geometry.padding.bottom - geometry.forecast[0].y}
              fill={product.accent === "green" ? "rgba(86, 127, 62, 0.08)" : "rgba(202, 45, 38, 0.08)"}
            />
          ) : null}
        </svg>
      </div>

      <div className="chart-legend">
        <div className="chart-legend__item">
          <span className={`chart-line ${product.accent === "green" ? "is-green" : "is-red"}`} />
          <span>Historical Trend</span>
        </div>
        <div className="chart-legend__item">
          <span className="chart-line is-delayed" />
          <span>Estimated During Data Delay</span>
        </div>
        <div className="chart-legend__item">
          <span
            className={`chart-line is-dashed ${product.accent === "green" ? "is-green" : "is-red"}`}
          />
          <span>AI Forecast</span>
        </div>
      </div>

      <div className="chart-note">
        Gold points show AI-estimated months used while official grocery price releases are still delayed.
      </div>

      <div className="chart-readout">
        <div className="chart-readout__label">Selected Point</div>
        <div className="chart-readout__value">
          {formatDate(active.date)} · C${active.price.toFixed(2)}
        </div>
        {active.forecast && active.date <= currentMonthKey ? (
          <div className="chart-readout__status">
            Estimated due to delayed official data release
          </div>
        ) : null}
        {active.forecast && active.lowerBound !== undefined && active.upperBound !== undefined ? (
          <div className="chart-readout__range">
            Range: C${active.lowerBound.toFixed(2)} - C${active.upperBound.toFixed(2)}
          </div>
        ) : null}
      </div>
    </section>
  );
}
