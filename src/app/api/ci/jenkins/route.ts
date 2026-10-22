import { NextRequest, NextResponse } from "next/server";
import { JenkinsClient } from "@/lib/jenkins";

const jenkins = new JenkinsClient("http://localhost:8089", "root", "root");

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      jobName = "TrainerMon-Pipeline",
      repoUrl = "https://github.com/Nawazwariya182/TrainerMon.git",
      branch = "main",
    } = body;

    // 1. Ensure pipeline job exists and points to repoUrl
    const provisionResult = await jenkins.ensurePipelineJob(jobName, repoUrl, branch);

    // 2. Trigger build execution
    const triggered = await jenkins.triggerBuild(jobName);

    return NextResponse.json({
      success: true,
      jobName,
      repoUrl,
      branch,
      provisioned: provisionResult,
      triggered,
      message: triggered
        ? `Successfully provisioned ${jobName} on Jenkins and triggered sequential pretraining pipeline.`
        : `Job ${jobName} ready, waiting in Jenkins build queue.`,
    });
  } catch (err: any) {
    console.error("Jenkins API error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to trigger Jenkins pipeline" },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const jobName = searchParams.get("jobName") || "TrainerMon-Pipeline";

    const latestBuild = await jenkins.getLatestBuild(jobName);
    const consoleLogs = await jenkins.getConsoleOutput(jobName, latestBuild?.number || "lastBuild");

    return NextResponse.json({
      success: true,
      jobName,
      connected: true,
      build: latestBuild,
      logs: consoleLogs.slice(-4000), // Return last 4000 characters for live streaming
    });
  } catch (err: any) {
    return NextResponse.json(
      { success: false, error: err.message, connected: false },
      { status: 500 }
    );
  }
}
