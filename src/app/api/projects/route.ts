import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "data");
const PROJECTS_FILE = path.join(DATA_DIR, "projects.json");

async function ensureFileExists() {
  try {
    await fs.mkdir(DATA_DIR, { recursive: true });
    await fs.access(PROJECTS_FILE);
  } catch {
    await fs.writeFile(PROJECTS_FILE, JSON.stringify([], null, 2), "utf-8");
  }
}

async function readProjects(): Promise<any[]> {
  await ensureFileExists();
  try {
    const raw = await fs.readFile(PROJECTS_FILE, "utf-8");
    return JSON.parse(raw || "[]");
  } catch {
    return [];
  }
}

async function writeProjects(projects: any[]) {
  await ensureFileExists();
  await fs.writeFile(PROJECTS_FILE, JSON.stringify(projects, null, 2), "utf-8");
}

export async function GET() {
  try {
    const projects = await readProjects();
    return NextResponse.json({ success: true, projects });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    if (!body.name || typeof body.name !== "string" || !body.name.trim()) {
      return NextResponse.json({ success: false, error: "Project name is required" }, { status: 400 });
    }

    const projects = await readProjects();
    const newProject = {
      id: crypto.randomUUID(),
      name: body.name.trim(),
      description: body.description?.trim() || body.idea?.trim() || "No description provided.",
      idea: body.idea?.trim() || body.description?.trim() || "",
      domains: Array.isArray(body.domains) ? body.domains : [],
      learningApproaches: Array.isArray(body.learningApproaches) ? body.learningApproaches : [],
      primaryGoal: body.primaryGoal || "Research / learning",
      model: body.model || "qwen3-8b",
      taskType: body.taskType || (Array.isArray(body.domains) && body.domains.length > 0 ? body.domains.join(", ") : "Inference"),
      maxTokens: body.maxTokens || 8192,
      contextWindow: body.contextWindow || 16384,
      status: "Ready",
      createdAt: new Date().toISOString(),
    };

    projects.unshift(newProject);
    await writeProjects(projects);

    return NextResponse.json({ success: true, project: newProject }, { status: 201 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const id = searchParams.get("id");
    if (!id) {
      return NextResponse.json({ success: false, error: "Missing project id" }, { status: 400 });
    }

    let projects = await readProjects();
    const initialLen = projects.length;
    projects = projects.filter((p) => p.id !== id);

    if (projects.length === initialLen) {
      return NextResponse.json({ success: false, error: "Project not found" }, { status: 404 });
    }

    await writeProjects(projects);
    return NextResponse.json({ success: true, message: "Project deleted successfully" });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
