import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";

const execAsync = promisify(exec);
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      projectId,
      repoUrl,
      branch = "main",
      commitMessage = "Initial commit: Pretraining pipeline from scratch",
      token = "",
    } = body;

    if (!projectId) {
      return NextResponse.json({ success: false, error: "Missing projectId" }, { status: 400 });
    }

    if (!repoUrl) {
      return NextResponse.json({ success: false, error: "Please provide a valid GitHub repository URL" }, { status: 400 });
    }

    const sanitizedProjectId = projectId.replace(/[^a-zA-Z0-9_-]/g, "_");
    const projectDir = path.join(WORKSPACES_ROOT, sanitizedProjectId);

    // Verify workspace directory exists
    try {
      await fs.access(projectDir);
    } catch {
      return NextResponse.json(
        { success: false, error: `Workspace folder not found at: ${projectDir}` },
        { status: 404 }
      );
    }

    // Format authenticated remote URL if token provided
    let targetRemoteUrl = repoUrl.trim();
    if (token.trim() && targetRemoteUrl.startsWith("https://")) {
      const cleanUrl = targetRemoteUrl.replace(/^https:\/\/[^@]+@/, "https://");
      const urlWithoutScheme = cleanUrl.replace(/^https:\/\//, "");
      targetRemoteUrl = `https://${encodeURIComponent(token.trim())}@${urlWithoutScheme}`;
    }

    const logs: string[] = [];

    // Helper to run git commands inside workspace
    const runGit = async (cmd: string) => {
      try {
        const { stdout, stderr } = await execAsync(cmd, { cwd: projectDir });
        const out = (stdout || "").trim();
        const err = (stderr || "").trim();
        if (out) logs.push(out);
        if (err) logs.push(err);
        return { success: true, stdout: out, stderr: err };
      } catch (e: any) {
        const errStr = (e.stderr || e.message || "").trim();
        logs.push(`[Error] ${errStr}`);
        return { success: false, error: errStr };
      }
    };

    // 1. Check if .git exists, if not initialize
    const gitDir = path.join(projectDir, ".git");
    let isInitialized = false;
    try {
      await fs.access(gitDir);
      isInitialized = true;
    } catch {
      isInitialized = false;
    }

    if (!isInitialized) {
      logs.push("Initializing new Git repository...");
      await runGit("git init");
    }

    // Configure user identity if not already set locally
    await runGit('git config user.name "ML-Train Studio"');
    await runGit('git config user.email "studio@mltrain.local"');

    // 2. Set default branch to main
    await runGit(`git branch -M ${branch}`);

    // 3. Stage files
    logs.push("Staging workspace files (respecting .gitignore)...");
    const addRes = await runGit("git add -A");
    if (!addRes.success) {
      return NextResponse.json({ success: false, error: addRes.error, logs }, { status: 500 });
    }

    // 4. Commit
    logs.push(`Committing changes: "${commitMessage}"...`);
    const commitRes = await runGit(`git commit -m "${commitMessage.replace(/"/g, '\\"')}"`);
    // git commit may return error if working tree is clean; check if that's the case
    if (!commitRes.success && !commitRes.error?.includes("nothing to commit")) {
      // not fatal if already committed
      logs.push("Working tree was already clean or previously committed.");
    }

    // 5. Configure remote origin
    logs.push("Configuring remote origin...");
    // Remove origin if already exists
    await runGit("git remote remove origin");
    const remoteRes = await runGit(`git remote add origin ${targetRemoteUrl}`);
    if (!remoteRes.success) {
      return NextResponse.json({ success: false, error: remoteRes.error, logs }, { status: 500 });
    }

    // 6. Push to remote
    logs.push(`Pushing to ${branch} branch on GitHub...`);
    // Try normal push first, if upstream differs push with -u
    const pushRes = await runGit(`git push -u origin ${branch} --force`);

    // Mask sensitive token in returned logs
    const sanitizedLogs = logs.map((line) =>
      token ? line.replace(new RegExp(token, "g"), "***TOKEN***") : line
    );

    if (!pushRes.success) {
      return NextResponse.json({
        success: false,
        error: `Git push failed: ${pushRes.error}`,
        logs: sanitizedLogs,
      }, { status: 400 });
    }

    return NextResponse.json({
      success: true,
      message: `Successfully pushed workspace to GitHub repository (${branch} branch)`,
      repoUrl: repoUrl.trim(),
      branch,
      logs: sanitizedLogs,
    });
  } catch (err: any) {
    console.error("Git push API error:", err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
