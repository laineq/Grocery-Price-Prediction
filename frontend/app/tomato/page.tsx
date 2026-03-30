import { DetailPage } from "@/components/detail-page";
import { getProductSummary } from "@/lib/data";

export default async function TomatoPage() {
  const product = await getProductSummary("tomato");
  return <DetailPage product={product} />;
}
