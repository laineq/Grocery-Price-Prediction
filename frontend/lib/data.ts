import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";

export type ProduceKey = "avocado" | "tomato";

export type AppOutputPoint = {
  date: string;
  price: number;
  forecast?: boolean;
  lower_bound?: number;
  upper_bound?: number;
};

export type AppOutputPayload = {
  product: ProduceKey;
  unit_label: string;
  prediction_month: string | null;
  forecast_price: number | null;
  change_pct: number | null;
  series: AppOutputPoint[];
};

export type PricePoint = {
  date: string;
  price: number;
  forecast?: boolean;
  lowerBound?: number;
  upperBound?: number;
};

export type ProductSummary = {
  key: ProduceKey;
  name: string;
  unitLabel: string;
  predictionMonth: string;
  forecastPrice: number;
  changePct: number;
  accent: "green" | "red";
  heroGradient: string;
  imageEmoji: string;
  detailTag: string;
  chart: PricePoint[];
  drivers: Array<{
    title: string;
    description: string;
    icon: "weather" | "exchange" | "gas" | "import" | "cpi";
  }>;
};

type ProductUiConfig = Omit<
  ProductSummary,
  "unitLabel" | "predictionMonth" | "forecastPrice" | "changePct" | "chart"
>;

const PRODUCT_UI: Record<ProduceKey, ProductUiConfig> = {
  avocado: {
    key: "avocado",
    name: "Avocado",
    accent: "green",
    heroGradient: "linear-gradient(160deg, #b7dc67 0%, #7ea844 100%)",
    imageEmoji: "🥑",
    detailTag: "Market Detail",
    drivers: [
      {
        title: "Temperature + Precipitation",
        description: "Michoacan + Jalisco + Estado de Mexico climatic monitoring.",
        icon: "weather",
      },
      {
        title: "Exchange Rate",
        description: "Fluctuations in MXN/CAD and USD/CAD currency pairs.",
        icon: "exchange",
      },
      {
        title: "Gas Price",
        description: "Diesel oil price trends across Canada and the United States.",
        icon: "gas",
      },
      {
        title: "Import Quantity",
        description: "Volume of avocado shipments originating from Mexico.",
        icon: "import",
      },
      {
        title: "Consumer Price Index (CPI)",
        description: "Macroeconomic inflation tracking for food and energy sectors.",
        icon: "cpi",
      },
    ],
  },
  tomato: {
    key: "tomato",
    name: "Tomato",
    accent: "red",
    heroGradient: "linear-gradient(160deg, #381313 0%, #781818 100%)",
    imageEmoji: "🍅",
    detailTag: "Market Detail",
    drivers: [
      {
        title: "Temperature + Precipitation",
        description: "Sinaloa in Mexico climatic monitoring.",
        icon: "weather",
      },
      {
        title: "Exchange Rate",
        description: "Fluctuations in MXN/CAD and USD/CAD currency pairs.",
        icon: "exchange",
      },
      {
        title: "Gas Price",
        description: "Diesel oil price trends across Canada and the United States.",
        icon: "gas",
      },
      {
        title: "Import Quantity",
        description: "Volume of tomato shipments originating from Mexico.",
        icon: "import",
      },
      {
        title: "Consumer Price Index (CPI)",
        description: "Macroeconomic inflation tracking for food and energy sectors.",
        icon: "cpi",
      },
    ],
  },
};

const APP_OUTPUT_BUCKET =
  process.env.APP_OUTPUT_BUCKET ?? process.env.BUCKET_NAME;
const APP_OUTPUT_PREFIX = (process.env.APP_OUTPUT_PREFIX ?? "app-output").replace(/\/+$/, "");
const AWS_REGION = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "ca-central-1";

let s3Client: S3Client | null = null;

function getS3Client() {
  if (!s3Client) {
    s3Client = new S3Client({ region: AWS_REGION });
  }
  return s3Client;
}

function normalizeUnitLabel(key: ProduceKey, unitLabel: string) {
  if (key === "avocado") {
    return "Pricing per unit";
  }

  if (unitLabel.toLowerCase() === "kg") {
    return "Pricing per kg";
  }

  return `Pricing per ${unitLabel}`;
}

function toPricePoint(point: AppOutputPoint): PricePoint {
  return {
    date: point.date,
    price: Number(point.price),
    forecast: Boolean(point.forecast),
    lowerBound:
      point.lower_bound === undefined ? undefined : Number(point.lower_bound),
    upperBound:
      point.upper_bound === undefined ? undefined : Number(point.upper_bound),
  };
}

function formatPredictionMonth(date: string) {
  return new Date(`${date}-01T00:00:00`).toLocaleDateString("en-CA", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}

function getNextMonthKey() {
  const now = new Date();
  const nextMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));
  const year = nextMonth.getUTCFullYear();
  const month = `${nextMonth.getUTCMonth() + 1}`.padStart(2, "0");
  return `${year}-${month}`;
}

function toProductSummary(payload: AppOutputPayload): ProductSummary {
  const ui = PRODUCT_UI[payload.product];
  const nextMonthKey = getNextMonthKey();
  const forecastPoints = payload.series.filter((point) => point.forecast);
  const selectedForecast =
    forecastPoints.find((point) => point.date === nextMonthKey) ??
    forecastPoints[0];
  const latestActual = [...payload.series]
    .filter((point) => !point.forecast)
    .sort((left, right) => left.date.localeCompare(right.date))
    .at(-1);
  const changePct =
    selectedForecast && latestActual && latestActual.price !== 0
      ? Number(
          (((Number(selectedForecast.price) - Number(latestActual.price)) /
            Number(latestActual.price)) *
            100).toFixed(1),
        )
      : 0;

  return {
    ...ui,
    unitLabel: normalizeUnitLabel(payload.product, payload.unit_label),
    predictionMonth: selectedForecast
      ? formatPredictionMonth(selectedForecast.date)
      : payload.prediction_month ?? "No prediction yet",
    forecastPrice: selectedForecast
      ? Number(selectedForecast.price)
      : payload.forecast_price ?? 0,
    changePct: selectedForecast ? changePct : payload.change_pct ?? 0,
    chart: payload.series.map(toPricePoint),
  };
}

async function readAppOutputFile(product: ProduceKey) {
  if (!APP_OUTPUT_BUCKET) {
    throw new Error(
      "Missing APP_OUTPUT_BUCKET or BUCKET_NAME for S3 app-output access.",
    );
  }

  const key = `${APP_OUTPUT_PREFIX}/${product}.json`;
  const response = await getS3Client().send(
    new GetObjectCommand({
      Bucket: APP_OUTPUT_BUCKET,
      Key: key,
    }),
  );

  const raw = await response.Body?.transformToString();
  if (!raw) {
    throw new Error(`Empty app-output payload for s3://${APP_OUTPUT_BUCKET}/${key}`);
  }

  return JSON.parse(raw) as AppOutputPayload;
}

export async function getProductSummary(product: ProduceKey) {
  const payload = await readAppOutputFile(product);
  return toProductSummary(payload);
}

export async function getDashboardProducts() {
  const products = await Promise.all([
    getProductSummary("avocado"),
    getProductSummary("tomato"),
  ]);
  return products;
}

export async function getRawAppOutput(product: ProduceKey) {
  return readAppOutputFile(product);
}
