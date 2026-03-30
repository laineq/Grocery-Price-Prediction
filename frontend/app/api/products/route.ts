import { NextResponse } from "next/server";
import { getRawAppOutput } from "@/lib/data";

export async function GET() {
  try {
    const [avocado, tomato] = await Promise.all([
      getRawAppOutput("avocado"),
      getRawAppOutput("tomato"),
    ]);

    return NextResponse.json({ avocado, tomato });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to load app-output files.",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
