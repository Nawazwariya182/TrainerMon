import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { JenkinsClient } from "@/lib/jenkins";

const execAsync = promisify(exec);
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");
const jenkins = new JenkinsClient("http://localhost:8089", "root", "root");

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      projectId = "5330b70a-8716-492c-aaf6-ea8dd6c9306b",
      repoUrl = "https://github.com/Nawazwariya182/TrainerMon.git",
      branch = "main",
      commitMessage = "feat: sync workspace pipeline and automated pretraining suite",
      triggerJenkins = true,
      jobName = "TrainerMon-Pipeline",
    } = body;

    const projectDir = path.join(WORKSPACES_ROOT, projectId.replace(/[^a-zA-Z0-9_-]/g, "_"));

    // 1. Stage, commit, and push in workspace directory
    try {
      await execAsync(`git add -A`, { cwd: projectDir });
      // Commit might return non-zero if working tree clean, so catch quietly
      try {
        await execAsync(`git commit -m "${commitMessage.replace(/"/g, '\\"')}"`, { cwd: projectDir });
      } catch (commitErr: any) {
        // Nothing to commit or already clean
      }
      await execAsync(`git push -u origin ${branch}`, { cwd: projectDir });
    } catch (gitErr: any) {
      console.error("Git push failed:", gitErr);
      return NextResponse.json(
        { success: false, error: `Git push failed: ${gitErr.message || gitErr}` },
        { status: 500 }
      );
    }

    // 2. Provision and trigger Jenkins CI/CD pipeline
    let jenkinsResult: any = null;
    if (triggerJenkins) {
      try {
        await jenkins.ensurePipelineJob(jobName, repoUrl, branch);
        const triggered = await jenkins.triggerBuild(jobName);
        jenkinsResult = {
          jobName,
          triggered,
          jenkinsUrl: `http://localhost:8089/job/${encodeURIComponent(jobName)}`,
        };
      } catch (jErr: any) {
        console.warn("Jenkins trigger warning:", jErr.message);
        jenkinsResult = { jobName, triggered: false, error: jErr.message };
      }
    }

    return NextResponse.json({
      success: true,
      pushed: true,
      repoUrl,
      branch,
      projectDir,
      jenkins: jenkinsResult,
      message: `Successfully pushed workspace to ${repoUrl} (${branch}) and initiated Jenkins pipeline build.`,
    });
  } catch (err: any) {
    console.error("Push API error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to push to GitHub" },
      { status: 500 }
    );
  }
}
