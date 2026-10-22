import { NextRequest, NextResponse } from "next/server";
import { JenkinsClient } from "@/lib/jenkins";
import fs from "fs/promises";
import path from "path";

const jenkins = new JenkinsClient("http://localhost:8089", "root", "root");
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");

// Store latest event status
let latestCiStatus: {
  lastEvent: string;
  buildNumber: number | string;
  jobName: string;
  timestamp: string;
  status: "SUCCESS" | "FAILED" | "HEALING" | "TUNING" | "SERVING_READY";
  details: string;
  previewUrl?: string;
} = {
  lastEvent: "IDLE",
  buildNumber: 0,
  jobName: "TrainerMon-Pipeline",
  timestamp: new Date().toISOString(),
  status: "SUCCESS",
  details: "Pipeline initialized.",
};

// Loop guard to prevent runaway build loops
let autoTuneCount = 0;
let lastAutoTuneTimestamp = 0;
const MAX_AUTO_TUNE_LOOPS = 1; // Prevent repeated cascading builds

export async function GET() {
  return NextResponse.json({ success: true, autoTuneCount, maxLoops: MAX_AUTO_TUNE_LOOPS, ...latestCiStatus });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      event = "PIPELINE_UPDATE",
      buildNumber = "lastBuild",
      jobName = "TrainerMon-Pipeline",
      projectId = "5330b70a-8716-492c-aaf6-ea8dd6c9306b",
    } = body;

    console.log(`[CI-Webhook] Received ${event} for ${jobName} #${buildNumber}`);

    const consoleLogs = await jenkins.getConsoleOutput(jobName, buildNumber);

    if (event === "PIPELINE_FAILURE") {
      const isQualityGateFailure =
        consoleLogs.includes("QUALITY_GATE_FAILED") ||
        consoleLogs.includes("below required 85.00%") ||
        consoleLogs.includes("exit code 2");

      if (isQualityGateFailure) {
        const now = Date.now();
        // Check if loop limit reached
        if (autoTuneCount >= MAX_AUTO_TUNE_LOOPS || (now - lastAutoTuneTimestamp < 120000)) {
          console.warn("[CI-Webhook] Auto-tune limit reached. Halting automated cascading trigger.");
          latestCiStatus = {
            lastEvent: "PIPELINE_FAILURE",
            buildNumber,
            jobName,
            timestamp: new Date().toISOString(),
            status: "FAILED",
            details: "Quality Gate violation (<85% Accuracy). Auto-tuning halted to prevent build loops. Code needs 95%+ architecture update.",
          };
          return NextResponse.json({
            success: true,
            action: "LOOP_PREVENTED",
            message: "Auto-tuning halted to avoid infinite build loops.",
          });
        }

        autoTuneCount++;
        lastAutoTuneTimestamp = now;

        latestCiStatus = {
          lastEvent: "PIPELINE_FAILURE",
          buildNumber,
          jobName,
          timestamp: new Date().toISOString(),
          status: "TUNING",
          details: "Quality Gate Failed (< 85% Accuracy). Invoking Autonomous Hyperparameter Tuner...",
        };

        // Trigger Auto-Tune API
        fetch("http://localhost:3000/api/ci/tune", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ projectId, jobName, buildNumber }),
        }).catch((err) => console.error("Auto-tune trigger error:", err));

        return NextResponse.json({
          success: true,
          action: "TUNING",
          message: "Quality Gate violation detected. Autonomous Hyperparameter Tuner dispatched.",
        });
      } else {
        // Traceback / Code error
        let failedScript = "train.py";
        if (consoleLogs.includes("download_datasets.py")) failedScript = "download_datasets.py";
        else if (consoleLogs.includes("eda_and_harmonization.py")) failedScript = "eda_and_harmonization.py";
        else if (consoleLogs.includes("evaluate_and_score.py")) failedScript = "evaluate_and_score.py";
        else if (consoleLogs.includes("visualize.py")) failedScript = "visualize.py";

        latestCiStatus = {
          lastEvent: "PIPELINE_FAILURE",
          buildNumber,
          jobName,
          timestamp: new Date().toISOString(),
          status: "HEALING",
          details: `Runtime failure detected in ${failedScript}. Invoking LLM Autonomous Code Healer...`,
        };

        // Trigger Self-Heal API
        fetch("http://localhost:3000/api/ci/heal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            projectId,
            scriptName: failedScript,
            errorLogs: consoleLogs.slice(-3000),
            jobName,
            buildNumber,
          }),
        }).catch((err) => console.error("Self-heal trigger error:", err));

        return NextResponse.json({
          success: true,
          action: "HEALING",
          failedScript,
          message: `Code failure detected in ${failedScript}. Autonomous Code Healer dispatched.`,
        });
      }
    } else if (event === "PIPELINE_SUCCESS") {
      latestCiStatus = {
        lastEvent: "PIPELINE_SUCCESS",
        buildNumber,
        jobName,
        timestamp: new Date().toISOString(),
        status: "SERVING_READY",
        details: "Quality Gate Passed (>=85% Accuracy). Synthesizing production serving web app & container...",
        previewUrl: "http://localhost:5000",
      };

      // 1. Synthesize serving files
      try {
        const synthRes = await fetch("http://localhost:3000/api/ci/synthesize-serving", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ projectId, port: 5000 }),
        });
        const synthData = await synthRes.json();

        // 2. Launch serving container / process
        await fetch("http://localhost:3000/api/ci/serving", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ projectId, port: 5000 }),
        });

        return NextResponse.json({
          success: true,
          action: "SERVING_DEPLOYED",
          synthesized: synthData.success,
          previewUrl: "http://localhost:5000",
          message: "Serving application synthesized and launched on http://localhost:5000",
        });
      } catch (synthErr: any) {
        console.error("Serving deployment error:", synthErr);
        return NextResponse.json({ success: false, error: synthErr.message }, { status: 500 });
      }
    }

    return NextResponse.json({ success: true, event });
  } catch (err: any) {
    console.error("Webhook processing error:", err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
