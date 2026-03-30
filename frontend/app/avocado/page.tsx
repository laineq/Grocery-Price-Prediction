import { DetailPage } from "@/components/detail-page";
import { getProductSummary } from "@/lib/data";

export default async function AvocadoPage() {
  const product = await getProductSummary("avocado");
  return <DetailPage product={product} />;
}
