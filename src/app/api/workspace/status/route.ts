import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const projectId = searchParams.get("projectId");

    if (!projectId) {
      return NextResponse.json({ success: false, error: "Missing projectId query parameter" }, { status: 400 });
    }

    const sanitizedProjectId = projectId.replace(/[^a-zA-Z0-9_-]/g, "_");
    const statusFilePath = path.join(WORKSPACES_ROOT, sanitizedProjectId, "EXECUTION_STATUS.json");

    try {
      const dataStr = await fs.readFile(statusFilePath, "utf-8");
      const statusData = JSON.parse(dataStr);
      return NextResponse.json({
        success: true,
        projectId: sanitizedProjectId,
        status: statusData,
      });
    } catch {
      return NextResponse.json({
        success: false,
        projectId: sanitizedProjectId,
        is_new: true,
        error: "EXECUTION_STATUS.json not yet initialized",
      });
    }
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
