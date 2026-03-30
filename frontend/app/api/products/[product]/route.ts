import { NextResponse } from "next/server";
import { getRawAppOutput, type ProduceKey } from "@/lib/data";

const VALID_PRODUCTS = new Set<ProduceKey>(["avocado", "tomato"]);

export async function GET(
  _request: Request,
  context: { params: Promise<{ product: string }> },
) {
  const { product } = await context.params;

  if (!VALID_PRODUCTS.has(product as ProduceKey)) {
    return NextResponse.json({ error: "Unknown product." }, { status: 404 });
  }

  try {
    const payload = await getRawAppOutput(product as ProduceKey);
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to load app-output file.",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
