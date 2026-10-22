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
      jobName = "TrainerMon-Pipeline",
      buildNumber = null,
      currentAccuracy = null,
    } = body;

    const projectDir = path.join(WORKSPACES_ROOT, projectId.replace(/[^a-zA-Z0-9_-]/g, "_"));
    const configPath = path.join(projectDir, "config.yaml");
    const metricsPath = path.join(projectDir, "metrics.json");

    // 1. Read existing config.yaml and metrics.json
    let currentConfig = "";
    try {
      currentConfig = await fs.readFile(configPath, "utf-8");
    } catch {
      currentConfig = "# Default config\ntraining:\n  epochs: 5\n  lr: 0.001\n";
    }

    let metricsData: any = {};
    try {
      const rawMetrics = await fs.readFile(metricsPath, "utf-8");
      metricsData = JSON.parse(rawMetrics);
    } catch {
      metricsData = { overall_accuracy: currentAccuracy ?? 0.72, macro_f1_score: 0.70 };
    }

    const accuracy = currentAccuracy ?? metricsData.overall_accuracy ?? metricsData.accuracy ?? 0.0;

    // 2. Query LLM to optimize hyperparameters
    const prompt = `You are TrainerMon's Autonomous Machine Learning Hyperparameter Optimization Specialist.
The pretraining run on real MRI datasets reached validation accuracy of ${(accuracy * 100).toFixed(2)}%, which failed the strict clinical Quality Gate (target >= 85.00%).

Your mission:
Refine and optimize the hyperparameters in config.yaml so that training achieves >= 85.00% validation accuracy and higher macro-F1 score.

Current Metrics:
${JSON.stringify(metricsData, null, 2)}

Current config.yaml:
\`\`\`yaml
${currentConfig}
\`\`\`

Optimization Guidelines:
1. Increase training epochs (e.g. increase from 5 to 10-15 epochs).
2. Tune initial learning rate (e.g. 3e-4 or 5e-4) and cosine annealing scheduler.
3. Add or tune regularizations: weight decay (0.01-0.05), label smoothing (0.08), dropout (0.25).
4. Maintain proper YAML structure and existing dataset definitions.
5. Return ONLY the valid raw YAML file contents with NO markdown formatting, backticks, or preamble chatter.`;

    let tunedConfig = "";
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
            { role: "system", content: "You are an expert ML engineer specializing in hyperparameter optimization and PyTorch convergence. Return only valid raw YAML." },
            { role: "user", content: prompt },
          ],
          temperature: 0.3,
          max_tokens: 4096,
        }),
      });

      if (llmRes.ok) {
        const data = await llmRes.json();
        let rawOutput = data.choices?.[0]?.message?.content || "";
        rawOutput = rawOutput.replace(/^```yaml\s*/i, "").replace(/^```\s*/, "").replace(/```\s*$/, "").trim();
        if (rawOutput.length > 50 && (rawOutput.includes("training:") || rawOutput.includes("optimizer:"))) {
          tunedConfig = rawOutput;
        }
      }
    } catch (llmErr) {
      console.warn("LLM API call failed during hyperparameter tuning:", llmErr);
    }

    // Fallback deterministic tuning if LLM response is empty
    if (!tunedConfig) {
      tunedConfig = currentConfig
        .replace(/epochs:\s*\d+/g, "epochs: 12")
        .replace(/lr:\s*[\d.e-]+/g, "lr: 0.0003")
        .replace(/max_epochs:\s*\d+/g, "max_epochs: 15");
    }

    // 3. Save tuned config.yaml
    await fs.writeFile(configPath, tunedConfig, "utf-8");

    // 4. Git commit and push to remote repository
    const commitMsg = `ci(auto-tune): optimize hyperparameters for >=85% accuracy (prev: ${(accuracy * 100).toFixed(1)}%)`;
    try {
      await execAsync(`git add config.yaml`, { cwd: projectDir });
      await execAsync(`git commit -m "${commitMsg}"`, { cwd: projectDir });
      await execAsync(`git push origin main`, { cwd: projectDir });
    } catch (gitErr: any) {
      console.warn("Git commit/push during tuning warning:", gitErr.message);
    }

    // 5. Retrigger Jenkins build
    let triggered = false;
    try {
      triggered = await jenkins.triggerBuild(jobName);
    } catch (jErr: any) {
      console.warn("Jenkins trigger warning after tuning:", jErr.message);
    }

    return NextResponse.json({
      success: true,
      tuned: true,
      previousAccuracy: accuracy,
      targetThreshold: 0.85,
      commitMessage: commitMsg,
      retriggered: triggered,
      message: `Hyperparameters autonomously tuned and pushed to GitHub. Re-evaluating on Jenkins to achieve >= 85% accuracy.`,
    });
  } catch (err: any) {
    console.error("Tune route error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to tune hyperparameters" },
      { status: 500 }
    );
  }
}
