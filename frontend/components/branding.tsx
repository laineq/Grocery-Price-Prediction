import Link from "next/link";
import { LeafIcon } from "@/components/icons";

export function BrandMark() {
  return (
    <Link href="/" style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
      <LeafIcon className="brand-icon" />
      <span className="brand-text">GroceryCast</span>
    </Link>
  );
}
