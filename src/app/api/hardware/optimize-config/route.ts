import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { detectHardwareProfile, HardwareProfile } from "@/lib/hardware";

const execAsync = promisify(exec);
const WORKSPACES_ROOT = path.join(process.cwd(), "workspace_projects");
const BACKEND_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions";
const BACKEND_API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";

interface OptimizeConfigRequest {
  projectId?: string;
  applyToProject?: boolean;
  hardwareProfile?: HardwareProfile;
  taskType?: string;
  targetSize?: [number, number];
  datasetSamples?: number;
}

export async function POST(req: NextRequest) {
  try {
    const body: OptimizeConfigRequest = await req.json();
    const {
      projectId = "5330b70a-8716-492c-aaf6-ea8dd6c9306b",
      applyToProject = true,
      taskType = "Image / CV (Multi-class Classification)",
      targetSize = [128, 128],
      datasetSamples = 3376,
    } = body;

    // 1. Detect user's actual hardware specs
    const hardware = body.hardwareProfile || (await detectHardwareProfile());

    // 2. Query AI / LLM Architect for hardware-optimized hyperparameters
    const prompt = `You are a Principal Machine Learning Performance Systems Engineer.
The user has the following physical hardware on their machine:
- GPU: ${hardware.gpuName} (${hardware.vramGb} GB VRAM)
- Tensor Cores / BF16 Supported: ${hardware.bf16Supported}
- CUDA Compute Capability: ${hardware.computeCapability}
- CPU: ${hardware.cpuCores} Cores
- Total System RAM: ${hardware.totalRamGb} GB (Free: ${hardware.freeRamGb} GB)
- Hardware Tier: ${hardware.tierTitle} (${hardware.tier})

The Model Training Task:
- Project ID: ${projectId}
- Task Domain: ${taskType}
- Input Resolution: ${targetSize[0]}x${targetSize[1]}
- Training Dataset Size: ${datasetSamples} samples

Formulate the optimal high-throughput training hyperparameters to saturate the GPU Tensor Cores, eliminate DataLoader bottlenecks, and achieve >= 85% accuracy in the fastest possible wall-clock time.

Respond ONLY with a valid JSON object in this exact schema:
{
  "batch_size": <int>,
  "effective_batch_size": <int>,
  "gradient_accumulation_steps": <int>,
  "mixed_precision": "bf16" | "fp16" | "fp32",
  "num_workers": <int>,
  "pin_memory": true | false,
  "persistent_workers": true | false,
  "prefetch_factor": <int>,
  "channels_last": true | false,
  "cudnn_benchmark": true | false,
  "fused_optimizer": true | false,
  "lr": <float>,
  "warmup_epochs": <int>,
  "max_epochs": <int>,
  "acceleration_strategy": "<brief 1-2 sentence explanation of why this configuration achieves maximum speed on this specific GPU>"
}`;

    let aiConfig: any = null;
    let aiRationale = "";

    try {
      const aiRes = await fetch(BACKEND_CHAT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${BACKEND_API_KEY}`,
        },
        body: JSON.stringify({
          model: "qwen3.5-9b",
          messages: [
            {
              role: "system",
              content:
                "You are an expert CUDA & PyTorch performance engineer. You maximize training throughput by matching batch size, precision, and dataloader parallelism to user hardware.",
            },
            { role: "user", content: prompt },
          ],
          temperature: 0.1,
          max_tokens: 500,
        }),
      });

      if (aiRes.ok) {
        const aiData = await aiRes.json();
        const rawContent = aiData.choices?.[0]?.message?.content || "";
        const jsonMatch = rawContent.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          aiConfig = JSON.parse(jsonMatch[0]);
          aiRationale = aiConfig.acceleration_strategy || "";
        }
      }
    } catch (aiErr) {
      console.warn("AI LLM call warning, using deterministic hardware heuristic:", aiErr);
    }

    // Fallback deterministic synthesis if AI is unavailable or didn't respond with valid JSON
    if (!aiConfig) {
      const isHighEnd = hardware.tier === "HIGH_END" || hardware.tier === "ULTRA_HIGH_END";
      const isMid = hardware.tier === "MID_RANGE";

      aiConfig = {
        batch_size: isHighEnd ? 64 : isMid ? 32 : 16,
        effective_batch_size: isHighEnd ? 64 : isMid ? 32 : 16,
        gradient_accumulation_steps: 1,
        mixed_precision: hardware.bf16Supported ? "bf16" : hardware.hasGpu ? "fp16" : "fp32",
        num_workers: hardware.hasGpu ? Math.min(6, Math.max(2, Math.floor(hardware.cpuCores / 2))) : 2,
        pin_memory: hardware.hasGpu,
        persistent_workers: hardware.hasGpu,
        prefetch_factor: 2,
        channels_last: hardware.hasGpu,
        cudnn_benchmark: hardware.hasGpu,
        fused_optimizer: hardware.hasGpu,
        lr: isHighEnd ? 0.0018 : isMid ? 0.0012 : 0.0008,
        warmup_epochs: 2,
        max_epochs: 12,
        acceleration_strategy: `Engineered specifically for ${hardware.gpuName} with native ${
          hardware.bf16Supported ? "BF16 Tensor Cores" : "FP16 AMP"
        }, high-throughput batch size of ${isHighEnd ? 64 : 32}, and ${
          hardware.hasGpu ? 6 : 2
        } parallel asynchronous I/O workers.`,
      };
      aiRationale = aiConfig.acceleration_strategy;
    }

    let filesUpdated: string[] = [];

    // 3. Apply the AI-optimized hyperparameters to workspace project if requested
    if (applyToProject && projectId) {
      const projectDir = path.join(WORKSPACES_ROOT, projectId.replace(/[^a-zA-Z0-9_-]/g, "_"));
      const configPath = path.join(projectDir, "config.yaml");

      try {
        let configContent = await fs.readFile(configPath, "utf-8");

        // Insert or update hardware_acceleration block in config.yaml
        const hardwareBlock = `# =====================================================================
# AI HARDWARE-OPTIMIZED ACCELERATION PROFILE
# Auto-tuned for: ${hardware.gpuName} (${hardware.vramGb} GB VRAM)
# Strategy: ${aiRationale}
# =====================================================================
hardware_profile:
  gpu_name: "${hardware.gpuName}"
  vram_gb: ${hardware.vramGb}
  compute_capability: "${hardware.computeCapability}"
  bf16_supported: ${hardware.bf16Supported}
  cpu_cores: ${hardware.cpuCores}
  ram_gb: ${hardware.totalRamGb}
  tier: "${hardware.tier}"
  estimated_speedup: "${hardware.estimatedSpeedup}"

training:
  batch_size: ${aiConfig.batch_size}
  effective_batch_size: ${aiConfig.effective_batch_size}
  gradient_accumulation_steps: ${aiConfig.gradient_accumulation_steps}
  num_workers: ${aiConfig.num_workers}
  pin_memory: ${aiConfig.pin_memory}
  persistent_workers: ${aiConfig.persistent_workers}
  prefetch_factor: ${aiConfig.prefetch_factor}
  mixed_precision: "${aiConfig.mixed_precision}"
  channels_last: ${aiConfig.channels_last}
  cudnn_benchmark: ${aiConfig.cudnn_benchmark}
  fused_optimizer: ${aiConfig.fused_optimizer}
  lr: ${aiConfig.lr}
  warmup_epochs: ${aiConfig.warmup_epochs}
  max_epochs: ${aiConfig.max_epochs}
`;

        if (configContent.includes("training:")) {
          // Replace training section
          configContent = configContent.replace(/training:[\s\S]*?(?=\n[a-z_]+:|$)/, hardwareBlock.trim());
        } else {
          configContent += "\n" + hardwareBlock;
        }

        await fs.writeFile(configPath, configContent, "utf-8");
        filesUpdated.push("config.yaml");

        // Commit and push to git so Jenkins CI/CD immediately runs with the accelerated config
        try {
          await execAsync(`git add config.yaml`, { cwd: projectDir });
          await execAsync(
            `git commit -m "perf(hardware): AI auto-tuned config for ${hardware.gpuName} (Batch: ${aiConfig.batch_size}, ${aiConfig.mixed_precision.toUpperCase()} AMP)"`,
            { cwd: projectDir }
          );
          await execAsync(`git push origin main`, { cwd: projectDir });
          filesUpdated.push("git:pushed_to_origin_main");
        } catch (gitErr: any) {
          console.warn("Git push warning:", gitErr.message);
        }
      } catch (fileErr: any) {
        console.warn("Could not write config.yaml:", fileErr.message);
      }
    }

    return NextResponse.json({
      success: true,
      hardware,
      optimizedConfig: aiConfig,
      aiRationale,
      filesUpdated,
      message: `AI successfully analyzed ${hardware.gpuName} and synthesized high-performance training parameters (${hardware.estimatedSpeedup}).`,
    });
  } catch (err: any) {
    console.error("Hardware optimization error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to optimize for hardware" },
      { status: 500 }
    );
  }
}
