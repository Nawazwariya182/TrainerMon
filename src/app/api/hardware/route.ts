import { NextRequest, NextResponse } from "next/server";
import { detectHardwareProfile } from "@/lib/hardware";

export async function GET(req: NextRequest) {
  try {
    const profile = await detectHardwareProfile();
    return NextResponse.json({
      success: true,
      profile,
    });
  } catch (err: any) {
    console.error("Hardware detection error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to detect hardware" },
      { status: 500 }
    );
  }
}
