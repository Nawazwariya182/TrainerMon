import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { exec, spawn } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");

// Check Docker daemon availability with a strict 2.5s timeout
async function checkDockerDaemon(): Promise<{ available: boolean; version?: string; error?: string }> {
  try {
    const { stdout } = await execAsync("docker version --format '{{.Server.Version}}'", { timeout: 2500 });
    const version = stdout.trim().replace(/['"]/g, "");
    if (version && !version.includes("error") && !version.includes("Cannot connect")) {
      return { available: true, version };
    }
  } catch (err: any) {
    return { available: false, error: err.message || "Docker daemon not reachable" };
  }
  return { available: false, error: "Docker daemon offline" };
}

// Check if port has an active listener and query health endpoint
async function checkServiceHealth(port: string | number) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (res.ok) {
      const data = await res.json();
      return { running: true, health: data };
    }
  } catch {
    // Offline
  }
  return { running: false, health: null };
}

// Keep active child process reference in memory
let activeChildProcess: any = null;
let currentRuntimeMode: "docker-container" | "native-python-process" | "offline" = "offline";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const port = searchParams.get("port") || "5000";

  const { running, health } = await checkServiceHealth(port);
  const docker = await checkDockerDaemon();

  // If service is running, detect whether it's running in Docker or native process
  let mode = currentRuntimeMode;
  if (running && mode === "offline") {
    try {
      const { stdout } = await execAsync("docker ps --filter name=trainermon-diagnostic-service --format '{{.Names}}'");
      if (stdout.includes("trainermon-diagnostic-service")) {
        mode = "docker-container";
      } else {
        mode = "native-python-process";
      }
    } catch {
      mode = "native-python-process";
    }
    currentRuntimeMode = mode;
  } else if (!running) {
    currentRuntimeMode = "offline";
    mode = "offline";
  }

  return NextResponse.json({
    running,
    port: parseInt(port),
    url: `http://localhost:${port}`,
    runtimeMode: mode,
    dockerAvailable: docker.available,
    dockerVersion: docker.version || null,
    dualModeStatus: docker.available
      ? "Docker daemon online & container ready"
      : "Docker daemon offline (native Python automatic fallback operational)",
    health,
  });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      projectId = "5330b70a-8716-492c-aaf6-ea8dd6c9306b",
      port = 5000,
      forceMode = "auto", // "auto" | "docker" | "native"
    } = body;

    const projectDir = path.join(WORKSPACES_ROOT, projectId.replace(/[^a-zA-Z0-9_-]/g, "_"));
    const servingDir = path.join(projectDir, "serving");

    // 1. Check if already responding
    const initialHealth = await checkServiceHealth(port);
    if (initialHealth.running && forceMode === "auto") {
      return NextResponse.json({
        success: true,
        running: true,
        runtimeMode: currentRuntimeMode,
        url: `http://localhost:${port}`,
        message: `Inference service already responding on port ${port} (${currentRuntimeMode})`,
        health: initialHealth.health,
      });
    }

    // 2. Automated Dual-Mode Evaluation
    const docker = await checkDockerDaemon();
    let launchedMode: "docker-container" | "native-python-process" = "native-python-process";
    let fallbackNotice: string | null = null;

    if (forceMode === "docker" && !docker.available) {
      fallbackNotice = "Docker daemon is stopped. Automatically falling back to native Python runtime.";
    }

    const shouldUseDocker = forceMode === "docker" || (forceMode === "auto" && docker.available);

    if (shouldUseDocker && docker.available) {
      try {
        console.log(`[Serving] Launching Docker container on port ${port}...`);
        await execAsync(`docker compose -f docker-compose.yml up -d --build`, { cwd: servingDir, timeout: 45000 });
        launchedMode = "docker-container";
        currentRuntimeMode = "docker-container";
      } catch (dockerErr: any) {
        console.warn(`[Serving] Docker launch failed: ${dockerErr.message}. Automatically falling back to native Python runtime.`);
        fallbackNotice = `Docker launch encountered error: ${dockerErr.message}. Seamlessly fell back to native Python runtime.`;
        launchedMode = "native-python-process";
      }
    }

    // If native python mode or fallback triggered
    if (launchedMode === "native-python-process") {
      console.log(`[Serving] Starting native Python Flask process on port ${port}...`);
      if (activeChildProcess) {
        try {
          activeChildProcess.kill();
        } catch {}
      }

      const child = spawn("python", ["app.py"], {
        cwd: servingDir,
        detached: true,
        stdio: "ignore",
        env: {
          ...process.env,
          PORT: String(port),
          PYTHONUNBUFFERED: "1",
        },
      });
      child.unref();
      activeChildProcess = child;
      currentRuntimeMode = "native-python-process";
    }

    // 3. Poll health check for up to 8 seconds
    let healthy = false;
    let healthData: any = null;
    for (let attempt = 0; attempt < 16; attempt++) {
      await new Promise((r) => setTimeout(r, 500));
      const chk = await checkServiceHealth(port);
      if (chk.running) {
        healthy = true;
        healthData = chk.health;
        break;
      }
    }

    return NextResponse.json({
      success: healthy,
      running: healthy,
      port,
      url: `http://localhost:${port}`,
      runtimeMode: currentRuntimeMode,
      dockerAvailable: docker.available,
      dockerVersion: docker.version || null,
      fallbackNotice,
      health: healthData,
      message: healthy
        ? `Serving engine is LIVE on http://localhost:${port} via ${currentRuntimeMode} (${docker.available ? "Docker active" : "Automatic native Python fallback"}).`
        : `Service process initiated via ${currentRuntimeMode}. Waiting for warm-up.`,
    });
  } catch (err: any) {
    console.error("Serving launch error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to start serving service" },
      { status: 500 }
    );
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const port = searchParams.get("port") || "5000";

    // 1. Stop docker container if running
    try {
      await execAsync("docker rm -f trainermon-diagnostic-service 2>&1");
    } catch {}

    // 2. Kill python process on port
    if (activeChildProcess) {
      try {
        activeChildProcess.kill();
        activeChildProcess = null;
      } catch {}
    }

    try {
      if (process.platform === "win32") {
        await execAsync(`powershell -Command "Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*app.py*' } | Stop-Process -Force"`);
      } else {
        await execAsync(`fuser -k ${port}/tcp`);
      }
    } catch {}

    currentRuntimeMode = "offline";
    return NextResponse.json({ success: true, stopped: true, port });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
