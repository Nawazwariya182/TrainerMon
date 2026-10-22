import { exec } from "child_process";
import { promisify } from "util";
import os from "os";

const execAsync = promisify(exec);
const BACKEND_HEALTH_URL = "http://127.0.0.1:8000/health";

export interface HardwareProfile {
  hasGpu: boolean;
  gpuName: string;
  vramGb: number;
  computeCapability: string;
  bf16Supported: boolean;
  cpuCores: number;
  totalRamGb: number;
  freeRamGb: number;
  platform: string;
  tier: "ULTRA_HIGH_END" | "HIGH_END" | "MID_RANGE" | "ENTRY_GPU" | "CPU_ONLY";
  tierTitle: string;
  recommendedBatchSize: number;
  recommendedWorkers: number;
  recommendedPrecision: "bf16" | "fp16" | "fp32";
  fusedOptimizerSupported: boolean;
  channelsLastSupported: boolean;
  estimatedSpeedup: string;
}

export async function detectHardwareProfile(): Promise<HardwareProfile> {
  const cpuCores = os.cpus().length || 8;
  const totalRamGb = Math.round((os.totalmem() / (1024 * 1024 * 1024)) * 10) / 10;
  const freeRamGb = Math.round((os.freemem() / (1024 * 1024 * 1024)) * 10) / 10;
  const platform = os.platform();

  let hasGpu = false;
  let gpuName = "CPU Fallback";
  let vramGb = 0;
  let computeCapability = "None";
  let bf16Supported = false;

  // 1. First probe FastAPI backend health which reads live nvidia-smi
  try {
    const res = await fetch(BACKEND_HEALTH_URL, { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      if (data.gpu && data.gpu.available) {
        hasGpu = true;
        gpuName = data.gpu.gpu_name || "NVIDIA GPU";
        vramGb = Math.round((data.gpu.memory_total_mb / 1024) * 10) / 10;
      }
    }
  } catch {
    // Proceed to direct Python probe
  }

  // 2. Direct PyTorch probe for tensor core details, compute capability & BF16
  try {
    const cmd = `python -c "import torch, json; info = {'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None', 'vram': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0, 'cap': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [], 'bf16': torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}; print(json.dumps(info))"`;
    const { stdout } = await execAsync(cmd, { timeout: 4000 });
    const pyInfo = JSON.parse(stdout.trim());
    if (pyInfo.cuda) {
      hasGpu = true;
      gpuName = pyInfo.device || gpuName;
      vramGb = pyInfo.vram || vramGb;
      bf16Supported = !!pyInfo.bf16;
      if (pyInfo.cap && pyInfo.cap.length >= 2) {
        computeCapability = `${pyInfo.cap[0]}.${pyInfo.cap[1]}`;
      }
    }
  } catch (pyErr) {
    console.warn("Direct PyTorch probe notice:", pyErr);
  }

  // 3. Classify Hardware Tier
  let tier: HardwareProfile["tier"] = "CPU_ONLY";
  let tierTitle = "CPU Only (Baseline)";
  let recommendedBatchSize = 16;
  let recommendedWorkers = Math.max(1, Math.min(4, Math.floor(cpuCores / 2)));
  let recommendedPrecision: "bf16" | "fp16" | "fp32" = "fp32";
  let fusedOptimizerSupported = false;
  let channelsLastSupported = false;
  let estimatedSpeedup = "1.0x (Standard CPU Execution)";

  if (hasGpu) {
    fusedOptimizerSupported = true;
    channelsLastSupported = true;

    if (vramGb >= 16) {
      tier = "ULTRA_HIGH_END";
      tierTitle = "Ultra High-End GPU (≥ 16GB VRAM)";
      recommendedBatchSize = 128;
      recommendedWorkers = Math.min(8, Math.floor(cpuCores * 0.75));
      recommendedPrecision = bf16Supported ? "bf16" : "fp16";
      estimatedSpeedup = "5.5x - 7.0x faster training";
    } else if (vramGb >= 10) {
      tier = "HIGH_END";
      tierTitle = "High-End Tensor Core GPU (10GB–16GB VRAM)";
      recommendedBatchSize = 64;
      recommendedWorkers = Math.min(6, Math.floor(cpuCores * 0.5));
      recommendedPrecision = bf16Supported ? "bf16" : "fp16";
      estimatedSpeedup = "3.8x - 4.5x faster training";
    } else if (vramGb >= 6) {
      tier = "MID_RANGE";
      tierTitle = "Mid-Range GPU (6GB–8GB VRAM)";
      recommendedBatchSize = 32;
      recommendedWorkers = Math.min(4, Math.floor(cpuCores * 0.4));
      recommendedPrecision = "fp16";
      estimatedSpeedup = "2.2x - 3.0x faster training";
    } else {
      tier = "ENTRY_GPU";
      tierTitle = "Entry GPU (≤ 4GB VRAM)";
      recommendedBatchSize = 16;
      recommendedWorkers = 2;
      recommendedPrecision = "fp16";
      estimatedSpeedup = "1.5x - 2.0x faster training";
    }
  }

  return {
    hasGpu,
    gpuName,
    vramGb,
    computeCapability,
    bf16Supported,
    cpuCores,
    totalRamGb,
    freeRamGb,
    platform,
    tier,
    tierTitle,
    recommendedBatchSize,
    recommendedWorkers,
    recommendedPrecision,
    fusedOptimizerSupported,
    channelsLastSupported,
    estimatedSpeedup,
  };
}
