import { DetailPage } from "@/components/detail-page";
import { products } from "@/lib/data";

export default function TomatoPage() {
  return <DetailPage product={products.tomato} />;
}
