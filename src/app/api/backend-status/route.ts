import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch("http://127.0.0.1:8000/health", {
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      return NextResponse.json({ online: false, error: "Backend returned non-200" });
    }
    const data = await res.json();
    return NextResponse.json({ online: true, ...data });
  } catch (err: any) {
    return NextResponse.json({
      online: false,
      message: "Backend inference engine offline or initializing",
    });
  }
}
