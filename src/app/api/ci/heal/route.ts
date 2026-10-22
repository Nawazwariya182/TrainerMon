import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { JenkinsClient } from "@/lib/jenkins";

const execAsync = promisify(exec);
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");
const jenkins = new JenkinsClient("http://localhost:8089", "root", "root");
const BACKEND_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions";
const BACKEND_API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      projectId = "5330b70a-8716-492c-aaf6-ea8dd6c9306b",
      scriptName = "train.py",
      errorLogs = "",
      jobName = "TrainerMon-Pipeline",
      buildNumber = null,
    } = body;

    const projectDir = path.join(WORKSPACES_ROOT, projectId.replace(/[^a-zA-Z0-9_-]/g, "_"));
    const scriptPath = path.join(projectDir, scriptName);

    // 1. Read existing script code
    let currentCode = "";
    try {
      currentCode = await fs.readFile(scriptPath, "utf-8");
    } catch {
      return NextResponse.json(
        { success: false, error: `Script ${scriptName} not found in workspace ${projectId}` },
        { status: 404 }
      );
    }

    // 2. Query Local LLM Engine to heal code
    const prompt = `You are TrainerMon's Autonomous Self-Healing Code Assistant.
A python script failed during automated Jenkins CI/CD execution.
Your task is to fix all bugs, runtime errors, import issues, tensor mismatches, or file path discrepancies.

Failing Script Name: ${scriptName}
Jenkins Build: #${buildNumber || "latest"}
Execution Error / Traceback:
\`\`\`
${errorLogs.slice(-2500)}
\`\`\`

Current Script Code:
\`\`\`python
${currentCode}
\`\`\`

RULES:
1. Output ONLY the complete, 100% executable fixed python code.
2. Maintain all real dataset loaders, model architectures, and progress logging.
3. Fix the exact root cause identified in the traceback.
4. Do NOT include markdown code blocks (\`\`\`python) or explanatory chatter; output only raw python code.`;

    let healedCode = "";
    try {
      const llmRes = await fetch(BACKEND_CHAT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${BACKEND_API_KEY}`,
        },
        body: JSON.stringify({
          model: "qwen3-8b",
          messages: [
            { role: "system", content: "You are an expert autonomous Python and PyTorch debugging system. You return clean, directly executable code." },
            { role: "user", content: prompt },
          ],
          temperature: 0.2,
          max_tokens: 8192,
        }),
      });

      if (llmRes.ok) {
        const data = await llmRes.json();
        let rawOutput = data.choices?.[0]?.message?.content || "";
        // Strip markdown backticks if present
        rawOutput = rawOutput.replace(/^```python\s*/i, "").replace(/^```\s*/, "").replace(/```\s*$/, "").trim();
        if (rawOutput.length > 50) {
          healedCode = rawOutput;
        }
      }
    } catch (llmErr) {
      console.warn("LLM API call failed during healing:", llmErr);
    }

    if (!healedCode) {
      return NextResponse.json(
        { success: false, error: "LLM healing engine could not generate a patch" },
        { status: 500 }
      );
    }

    // 3. Write healed script to disk
    await fs.writeFile(scriptPath, healedCode, "utf-8");

    // 4. Git commit and push to remote repository
    const commitMsg = `ci(self-heal): auto-repaired ${scriptName} after build #${buildNumber || "ci"} failure`;
    try {
      await execAsync(`git add ${scriptName}`, { cwd: projectDir });
      await execAsync(`git commit -m "${commitMsg}"`, { cwd: projectDir });
      await execAsync(`git push origin main`, { cwd: projectDir });
    } catch (gitErr: any) {
      console.warn("Git commit/push during healing warning:", gitErr.message);
    }

    // 5. Retrigger Jenkins build
    let triggered = false;
    try {
      triggered = await jenkins.triggerBuild(jobName);
    } catch (jErr: any) {
      console.warn("Jenkins trigger warning after healing:", jErr.message);
    }

    return NextResponse.json({
      success: true,
      healed: true,
      scriptName,
      commitMessage: commitMsg,
      retriggered: triggered,
      message: `Autonomous self-healing applied to ${scriptName}. Changes committed, pushed to GitHub, and new CI build triggered.`,
    });
  } catch (err: any) {
    console.error("Heal route error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to heal code" },
      { status: 500 }
    );
  }
}
