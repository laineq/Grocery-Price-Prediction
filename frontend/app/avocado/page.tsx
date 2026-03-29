import { DetailPage } from "@/components/detail-page";
import { products } from "@/lib/data";

export default function AvocadoPage() {
  return <DetailPage product={products.avocado} />;
}
