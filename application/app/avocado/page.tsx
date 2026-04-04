import { DetailPage } from "@/components/detail-page";
import { getProductSummary } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function AvocadoPage() {
  const product = await getProductSummary("avocado");
  return <DetailPage product={product} />;
}
