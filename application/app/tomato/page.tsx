import { DetailPage } from "@/components/detail-page";
import { getProductSummary } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function TomatoPage() {
  const product = await getProductSummary("tomato");
  return <DetailPage product={product} />;
}
