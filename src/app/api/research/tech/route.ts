import { NextRequest, NextResponse } from "next/server";

const API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";
const BACKEND_RAG_INDEX_URL = "http://127.0.0.1:8000/v1/research/rag/index";

export interface TechResearchPlan {
  title: string;
  architecture: {
    selected_name: string;
    rationale: string;
    layer_breakdown: string[];
    alternatives_evaluated: { name: string; pros: string; cons: string }[];
  };
  preprocessing: {
    steps: string[];
    spatial_resolution: string;
    normalization_formula: string;
    artifact_mitigation: string;
  };
  loss_formulation: {
    primary_loss: string;
    formula_description: string;
    class_weights: string;
    deep_supervision_weights: string;
  };
  hardware_and_compute: {
    estimated_parameters: string;
    estimated_vram_footprint: string;
    recommended_batch_size: string;
    gradient_accumulation_steps: string;
    mixed_precision: string;
    estimated_epochs_and_duration: string;
  };
  hyperparameters: {
    optimizer: string;
    base_learning_rate: string;
    warmup_schedule: string;
    weight_decay: string;
    patch_tile_size: string;
    validation_frequency: string;
  };
  implementation_todos: {
    id: number;
    title: string;
    description: string;
    status: "pending" | "in_progress" | "completed";
  }[];
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      project = {},
      selectedDatasets = [],
      answers = {},
      model = "qwen3-8b",
    } = body;

    const projectId = project.id || "proj_" + Math.random().toString(36).substring(2, 9);
    const projectName = project.name || "Machine Learning Project";
    const idea = project.idea || project.description || "";
    const domains: string[] = Array.isArray(project.domains) ? project.domains : [project.domains || "Image / CV"];

    const textCorpus = `${projectName} ${idea} ${domains.join(" ")}`.toLowerCase();
    const isMedical = textCorpus.includes("mri") || textCorpus.includes("tumor") || textCorpus.includes("brain") || textCorpus.includes("medical") || textCorpus.includes("lesion");
    const isTimeSeries = textCorpus.includes("time series") || textCorpus.includes("forecast") || textCorpus.includes("stock") || textCorpus.includes("sensor") || textCorpus.includes("energy");
    const isAudio = textCorpus.includes("audio") || textCorpus.includes("speech") || textCorpus.includes("voice");

    const primaryDataset = selectedDatasets.length > 0 ? selectedDatasets[0] : null;
    const datasetName = primaryDataset ? primaryDataset.title : isMedical ? "BraTS 2024 Adult Glioma Dataset" : isTimeSeries ? "NREL Solar Power 5-Min Dataset" : "Standard Domain Corpus";

    let plan: TechResearchPlan;

    if (isMedical) {
      plan = {
        title: `Foundation Pretraining & Architecture Blueprint: ${projectName}`,
        architecture: {
          selected_name: "3D Residual U-Net (nnU-Net Dynamic Architecture with Deep Supervision)",
          rationale: `Tailored for volumetric multi-parametric MRI segmentation on ${datasetName}. When training from scratch with random weight initialization, residual 3D convolutions with InstanceNorm3d avoid the severe gradient vanishing common in standard UNet backbones, while deep supervision injects auxiliary gradients across multi-resolution encoder levels.`,
          layer_breakdown: [
            "Input Tensor: (B, 4, 128, 128, 128) - 4 Channels: T1, T1Gd, T2, FLAIR",
            "Encoder Stage 1: 2x Conv3D(32 channels, 3x3x3), InstanceNorm3d, LeakyReLU(0.01), Stride 1",
            "Encoder Stage 2: Residual Conv3D(64 channels, 3x3x3), Stride 2 downsampling",
            "Encoder Stage 3: Residual Conv3D(128 channels, 3x3x3), Stride 2 downsampling",
            "Encoder Stage 4: Residual Conv3D(256 channels, 3x3x3), Stride 2 downsampling",
            "Bottleneck: Residual Conv3D(320 channels, 3x3x3) with 0.2 Dropout",
            "Decoder Stages: Transposed Conv3D with Skip Connections and Concatenation",
            "Multi-Scale Deep Supervision Heads: Softmax outputs at 1/1, 1/2, and 1/4 resolutions",
          ],
          alternatives_evaluated: [
            {
              name: "Swin UNETR (Hierarchical Vision Transformer)",
              pros: "Excellent long-range global context across distant 3D slices via shifted window self-attention.",
              cons: "Requires large pre-training token/patch volume; prone to overfitting on smaller clinical cohorts under 2,000 subjects.",
            },
            {
              name: "3D ConvNeXt-V2 (Pure Convolutions)",
              pros: "Modern depthwise separable 3D convolutions with large 7x7x7 receptive fields.",
              cons: "Higher memory utilization during dense multi-resolution skip connection concatenation.",
            },
          ],
        },
        preprocessing: {
          steps: [
            "1. Co-registration of multi-sequence MRI series (T1, T1Gd, T2, FLAIR) to SRI24 anatomical space",
            "2. Skull-Stripping / Brain Extraction using consensus Brain Extraction Tool (BET)",
            "3. N4 Bias Field Inhomogeneity Correction across all 4 MRI sequence channels",
            "4. Isotropic spatial resampling to 1.0 x 1.0 x 1.0 mm³ voxel spacing via 3rd-order B-Spline",
            "5. Brain-tissue-masked Z-score intensity standardization (zero mean, unit variance ignoring non-brain background voxels)",
          ],
          spatial_resolution: "1.0 x 1.0 x 1.0 mm³ isotropic voxel grid (Target patch size: 128 x 128 x 128)",
          normalization_formula: "z = (I - mean(I_brain)) / (std(I_brain) + 1e-7)",
          artifact_mitigation: "N4 Bias Field Correction + Percentile intensity clipping at [0.5%, 99.5%] to eliminate magnetic susceptibility artifacts.",
        },
        loss_formulation: {
          primary_loss: "Hybrid Combo Loss: Soft Dice Loss + Focal Cross-Entropy Loss",
          formula_description: "L_total = λ_dice * L_Dice(P, Y) + λ_focal * L_Focal(P, Y, γ=2.0) across Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) nested regions.",
          class_weights: "Inverse square root of voxel class frequencies (Background: 0.1, Non-enhancing: 1.0, Edema: 1.2, Enhancing: 2.5)",
          deep_supervision_weights: "Full resolution: 0.5, Half resolution (1/2): 0.3, Quarter resolution (1/4): 0.2",
        },
        hardware_and_compute: {
          estimated_parameters: "28.4 Million Trainable Parameters (From Scratch)",
          estimated_vram_footprint: "Approx. 8.4 GB to 10.2 GB working memory for (B=2, C=4, 128x128x128) 3D sub-volume patches under AMP BF16",
          recommended_batch_size: "Batch size = 2 per device (with gradient accumulation of 4 steps for effective batch size 8)",
          gradient_accumulation_steps: "4 steps",
          mixed_precision: "Automatic Mixed Precision (AMP BF16 or FP16) with GradScaler to prevent underflow",
          estimated_epochs_and_duration: "150 Epochs (~2.5 to 3.5 hours on modern local accelerators)",
        },
        hyperparameters: {
          optimizer: "AdamW (lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999))",
          base_learning_rate: "1e-4 with 5-epoch linear warmup",
          warmup_schedule: "Cosine Annealing with minimum learning rate 1e-6",
          weight_decay: "0.01 (L2 regularization on convolutional kernel weights)",
          patch_tile_size: "128 x 128 x 128 sub-volume (50% overlapping foreground-biased sampling)",
          validation_frequency: "Every 2 epochs (Dice Similarity Coefficient & 95% Hausdorff Distance)",
        },
        implementation_todos: [
          {
            id: 1,
            title: "Download & Verify Dataset Scans",
            description: `Acquire and extract verified NIfTI archives for ${datasetName}; verify 4 sequence channels per subject.`,
            status: "pending",
          },
          {
            id: 2,
            title: "Build Preprocessing & N4 Bias Pipeline",
            description: "Implement skull-stripping mask extraction, N4 bias field correction, and 1mm³ isotropic resampling.",
            status: "pending",
          },
          {
            id: 3,
            title: "Generate Foreground-Biased 3D Patches",
            description: "Construct dynamic patch sampler that ensures at least 66% of training sub-volumes contain tumor voxels.",
            status: "pending",
          },
          {
            id: 4,
            title: "Construct 3D Residual U-Net Backbone",
            description: "Assemble encoder, bottleneck, decoder, and deep supervision heads with Kaiming Normal weight initialization.",
            status: "pending",
          },
          {
            id: 5,
            title: "Implement Soft Dice + Focal Combo Loss",
            description: "Formulate numerically stable Soft Dice calculation with smooth=1.0 and focal modulation factor γ=2.0.",
            status: "pending",
          },
          {
            id: 6,
            title: "Execute Training from Scratch with AMP BF16",
            description: "Run 150-epoch training loop with Cosine Annealing, tracking validation DSC across WT, TC, and ET.",
            status: "pending",
          },
          {
            id: 7,
            title: "Implement Sliding Window Inference & Export",
            description: "Build Gaussian-weighted overlapping sliding window stitching and export final weights to ONNX Runtime FP16.",
            status: "pending",
          },
        ],
      };
    } else if (isTimeSeries) {
      plan = {
        title: `Foundation Pretraining & Architecture Blueprint: ${projectName}`,
        architecture: {
          selected_name: "PatchTST (Patch Time Series Transformer with Channel Independence)",
          rationale: `Designed for multi-horizon forecasting on ${datasetName}. Sub-series patching segments temporal continuity into localized semantic tokens, reducing attention complexity from O(L²) to O((L/P)²) while channel independence prevents inter-series cross-noise.`,
          layer_breakdown: [
            "Input Horizon: Lookback L=512 steps, Forecast H=96 steps",
            "Patching Layer: Patch length P=16, Stride S=8 (Generates N=63 patch tokens)",
            "Linear Projection: Maps patches to d_model=128 dimensions",
            "Positional Encoding: Learnable 1D additive positional embeddings",
            "Transformer Encoder: 4 Layers, 8 Attention Heads, GELU activation, d_ff=256",
            "Head: Flatten and Linear Projection head directly to H=96 forecast outputs",
          ],
          alternatives_evaluated: [
            {
              name: "TimesNet (2D Temporal Variations)",
              pros: "Transforms 1D series into 2D tensors via FFT to capture overlapping cyclic intra-period variations.",
              cons: "Slightly higher computation latency for high-frequency multivariate series.",
            },
            {
              name: "DLinear / NLinear",
              pros: "Extremely lightweight linear decomposition into trend and seasonal components.",
              cons: "Lacks non-linear attention capability on complex multi-sensor cross-dependencies.",
            },
          ],
        },
        preprocessing: {
          steps: [
            "1. Temporal continuity validation and missing observation forward-fill interpolation",
            "2. Chronological calendar feature extraction (Hour-of-day, Day-of-week, Month sine/cosine encodings)",
            "3. Reversible Instance Normalization (RevIN) to eliminate distribution shift and stationarity drift",
          ],
          spatial_resolution: "Lookback window L=512 steps, Patch length P=16, Stride S=8",
          normalization_formula: "RevIN: x_norm = (x - mean(x)) / (std(x) + eps); y_pred = y_norm * std(x) + mean(x)",
          artifact_mitigation: "Outlier clipping via rolling median filter with 3-sigma tolerance.",
        },
        loss_formulation: {
          primary_loss: "Quantile Pinball Loss (10th, 50th, 90th percentiles) + MSE",
          formula_description: "L_pinball(y, y_hat, q) = max(q * (y - y_hat), (1 - q) * (y_hat - y)) providing calibrated uncertainty intervals.",
          class_weights: "q = [0.10, 0.50, 0.90] with equal weighting (1/3 each)",
          deep_supervision_weights: "None (Direct multi-horizon projection)",
        },
        hardware_and_compute: {
          estimated_parameters: "1.2 Million Trainable Parameters",
          estimated_vram_footprint: "Approx. 1.8 GB to 2.4 GB working memory for batch size 64",
          recommended_batch_size: "Batch size = 64 (High throughput)",
          gradient_accumulation_steps: "1 step",
          mixed_precision: "Automatic Mixed Precision (AMP FP16)",
          estimated_epochs_and_duration: "50 Epochs (~20 to 35 minutes on modern local accelerators)",
        },
        hyperparameters: {
          optimizer: "AdamW (lr=3e-4, weight_decay=1e-3)",
          base_learning_rate: "3e-4 with 5-epoch linear warmup",
          warmup_schedule: "Cosine Annealing schedule",
          weight_decay: "0.001",
          patch_tile_size: "P=16, S=8",
          validation_frequency: "Every epoch (MAE, MSE, and Quantile Coverage)",
        },
        implementation_todos: [
          {
            id: 1,
            title: "Format Chronological Sliding Windows",
            description: "Segment raw time series into (L=512, H=96) lookback and target horizons.",
            status: "pending",
          },
          {
            id: 2,
            title: "Implement RevIN Layer",
            description: "Build reversible instance normalization module with learnable affine scaling.",
            status: "pending",
          },
          {
            id: 3,
            title: "Construct Patch Time Series Transformer",
            description: "Assemble channel-independent patch embedding and multi-head self-attention layers.",
            status: "pending",
          },
          {
            id: 4,
            title: "Train with Quantile Loss & Evaluate Walk-Forward Splits",
            description: "Optimize across 50 epochs with rolling origin walk-forward cross-validation.",
            status: "pending",
          },
        ],
      };
    } else {
      // General Vision / General ML fallback
      plan = {
        title: `Foundation Pretraining & Architecture Blueprint: ${projectName}`,
        architecture: {
          selected_name: "ConvNeXt-V2 (Modern Pure-Convolutional Backbone)",
          rationale: `Selected for ${datasetName}. Combines inverted bottleneck blocks and depthwise 7x7 convolutions with global response normalization (GRN), achieving transformer-level representations with fast, stable convolutional training from random initialization.`,
          layer_breakdown: [
            "Input: (B, 3, 512, 512) Color RGB Images",
            "Stem: Conv2D(96 channels, 4x4, stride 4) + LayerNorm",
            "Stage 1: 3x ConvNeXt Blocks (96 channels)",
            "Stage 2: 3x ConvNeXt Blocks (192 channels, stride 2)",
            "Stage 3: 9x ConvNeXt Blocks (384 channels, stride 2)",
            "Stage 4: 3x ConvNeXt Blocks (768 channels, stride 2)",
            "Head: Global Average Pooling + LayerNorm + Linear Classifier / Projection Head",
          ],
          alternatives_evaluated: [
            {
              name: "Vision Transformer (ViT-B/16)",
              pros: "Global self-attention across image patches without inductive bias constraints.",
              cons: "Requires heavy data augmentation and high token count to converge from scratch.",
            },
            {
              name: "ResNet-50 Modernized",
              pros: "Well understood residual baseline with fast training steps.",
              cons: "Lower parameter efficiency and smaller effective receptive fields than ConvNeXt.",
            },
          ],
        },
        preprocessing: {
          steps: [
            "1. Aspect-ratio preserving bilinear resizing to 512x512 with letterbox padding",
            "2. ImageNet-standardized RGB mean subtraction and variance normalization",
            "3. MixUp (alpha=0.8) and CutMix (alpha=1.0) augmentation batches",
          ],
          spatial_resolution: "512 x 512 RGB",
          normalization_formula: "x_norm = (x / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]",
          artifact_mitigation: "Random erasing and color jittering to eliminate camera bias.",
        },
        loss_formulation: {
          primary_loss: "Focal Cross-Entropy Loss (gamma=2.0) with Label Smoothing (epsilon=0.1)",
          formula_description: "L_focal = -alpha * (1 - pt)^gamma * log(pt)",
          class_weights: "Uniform with label smoothing 0.1",
          deep_supervision_weights: "None",
        },
        hardware_and_compute: {
          estimated_parameters: "27.8 Million Trainable Parameters",
          estimated_vram_footprint: "Approx. 5.6 GB to 7.2 GB for batch size 16 under AMP BF16",
          recommended_batch_size: "Batch size = 16 (Gradient accumulation: 2 steps for effective batch 32)",
          gradient_accumulation_steps: "2 steps",
          mixed_precision: "Automatic Mixed Precision (AMP BF16)",
          estimated_epochs_and_duration: "100 Epochs (~1.5 to 2.2 hours on modern local accelerators)",
        },
        hyperparameters: {
          optimizer: "AdamW (lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))",
          base_learning_rate: "5e-4 with 10-epoch linear warmup",
          warmup_schedule: "Cosine Decay schedule to 1e-6",
          weight_decay: "0.05",
          patch_tile_size: "512 x 512",
          validation_frequency: "Every epoch (Top-1 Accuracy and Macro F1)",
        },
        implementation_todos: [
          {
            id: 1,
            title: "Prepare Data Loaders & Augmentations",
            description: "Implement RandAugment, CutMix, and MixUp data pipeline.",
            status: "pending",
          },
          {
            id: 2,
            title: "Construct ConvNeXt-V2 Backbone",
            description: "Assemble inverted bottleneck stages with Global Response Normalization (GRN).",
            status: "pending",
          },
          {
            id: 3,
            title: "Execute Training Loop from Scratch",
            description: "Train across 100 epochs with AdamW, Cosine Warmup, and AMP BF16.",
            status: "pending",
          },
          {
            id: 4,
            title: "Validate and Export Checkpoints",
            description: "Evaluate top-1 validation accuracy and export model to ONNX FP16 runtime.",
            status: "pending",
          },
        ],
      };
    }

    // Automatically index technical research into ChromaDB Vector Store
    let indexedIntoRag = false;
    try {
      const ragRes = await fetch(BACKEND_RAG_INDEX_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${API_KEY}`,
        },
        body: JSON.stringify({
          project_id: projectId,
          tech_research: plan,
        }),
      });
      if (ragRes.ok) {
        indexedIntoRag = true;
      }
    } catch (ragErr) {
      console.warn("Could not index technical research into ChromaDB:", ragErr);
    }

    return NextResponse.json({
      success: true,
      project_id: projectId,
      research: plan,
      indexed_into_rag: indexedIntoRag,
    });
  } catch (err: any) {
    console.error("Technical research generation error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to generate technical research" },
      { status: 500 }
    );
  }
}
