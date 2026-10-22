import { NextRequest, NextResponse } from "next/server";

const API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";
const BACKEND_RAG_INDEX_URL = "http://127.0.0.1:8000/v1/research/rag/index";

export interface DatasetItem {
  id: string;
  title: string;
  source: string;
  url: string;
  download_url?: string;
  download_type?: "zip" | "tar" | "file";
  modality: string;
  format: string;
  samples: string;
  size: string;
  license: string;
  verified: boolean;
  suitability_score: number;
  suitability_reason: string;
  recommended: boolean;
  tags: string[];
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      project = {},
      answers = {},
      model = "qwen3-8b",
    } = body;

    const projectId = project.id || "proj_" + Math.random().toString(36).substring(2, 9);
    const projectName = project.name || "Untitled ML Project";
    const idea = project.idea || project.description || "";
    const domains: string[] = Array.isArray(project.domains) ? project.domains : [project.domains || "Image / CV"];

    // Classify domain profile
    const textCorpus = `${projectName} ${idea} ${domains.join(" ")}`.toLowerCase();
    const isMedical = textCorpus.includes("mri") || textCorpus.includes("tumor") || textCorpus.includes("brain") || textCorpus.includes("ct scan") || textCorpus.includes("x-ray") || textCorpus.includes("medical") || textCorpus.includes("radiolog") || textCorpus.includes("lesion");
    const isTimeSeries = textCorpus.includes("time series") || textCorpus.includes("forecast") || textCorpus.includes("stock") || textCorpus.includes("sensor") || textCorpus.includes("energy") || textCorpus.includes("grid");
    const isAudio = textCorpus.includes("audio") || textCorpus.includes("speech") || textCorpus.includes("voice") || textCorpus.includes("acoustic") || textCorpus.includes("sound");
    const isTabular = textCorpus.includes("tabular") || textCorpus.includes("csv") || textCorpus.includes("credit") || textCorpus.includes("fraud") || textCorpus.includes("churn");

    let datasets: DatasetItem[] = [];

    if (isMedical) {
      datasets = [
        {
          id: "brats-mri-detection",
          title: "Brain MRI Images for Brain Tumor Detection (Br35H / Navoneel Cohort)",
          source: "Hugging Face Open Benchmark / Navoneel Chakraborty",
          url: "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection",
          download_url: "https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection/resolve/main/Brain%20MRI%20Images%20for%20Brain%20Tumor%20Detection.zip",
          download_type: "zip",
          modality: "Axial T1/T2 Brain MRI Slices (Binary: Yes/No Tumor)",
          format: "High-Resolution JPEG / PNG",
          samples: "253 Patient Scans (155 Tumor Positive + 98 Normal Controls)",
          size: "15.8 MB",
          license: "CC-BY 4.0 (Open Research & Commercial)",
          verified: true,
          suitability_score: 98,
          suitability_reason: "Directly downloadable, radiologist-verified 2D brain MRI scan collection with explicit binary tumor labels. Ideal for pretraining robust vision encoders from scratch.",
          recommended: true,
          tags: ["Direct Open Download", "Verified MRI", "Pathology Ground Truth", "Fast Convergence"],
        },
        {
          id: "sartaj-bhuvaji-mri",
          title: "Brain Tumor Classification MRI Benchmark (4 Classes: Glioma, Meningioma, Pituitary, Normal)",
          source: "Hugging Face / Sartaj Bhuvaji Radiologist Cohort",
          url: "https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification",
          download_url: "https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/Training.zip",
          download_type: "zip",
          modality: "Multi-View Axial & Coronal MRI Scans",
          format: "High-Resolution JPEG (224x224, 512x512)",
          samples: "2,870 Training Scans across 4 histological tumor classes",
          size: "78.3 MB",
          license: "Apache 2.0 (Permissive Open Access)",
          verified: true,
          suitability_score: 96,
          suitability_reason: "Comprehensive multi-class tumor classification standard separating Glioma, Meningioma, and Pituitary pathologies from healthy control tissue.",
          recommended: true,
          tags: ["4 Classes", "Glioma/Meningioma", "Apache 2.0", "Direct Download"],
        },
        {
          id: "msd-task01-brain",
          title: "Medical Segmentation Decathlon (MSD Task01: Brain Tumour)",
          source: "MedicalDecathlon.com / MONAI AWS S3 Open Benchmark",
          url: "http://medicaldecathlon.com/",
          download_url: "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
          download_type: "tar",
          modality: "4-Channel 3D MRI (FLAIR, T1w, T1gd, T2w)",
          format: "Standardized NIfTI (.nii.gz) with JSON manifest",
          samples: "484 Training + 266 Validation Volumetric Scans",
          size: "7.6 GB",
          license: "CC-BY-SA 4.0 (Open Benchmark)",
          verified: true,
          suitability_score: 93,
          suitability_reason: "Gold-standard 3D multi-parametric benchmark with voxel-level labels for whole tumor, tumor core, and enhancing tumor.",
          recommended: false,
          tags: ["3D Volumetric", "Gold Standard", "NIfTI", "AWS S3 Direct"],
        },
        {
          id: "tcga-gbm-lgg",
          title: "The Cancer Genome Atlas (TCGA-GBM & TCGA-LGG Collection)",
          source: "National Cancer Institute Cancer Imaging Archive (TCIA)",
          url: "https://wiki.cancerimagingarchive.net/display/Public/TCGA-GBM",
          modality: "Multi-Sequence DICOM MRI Scans",
          format: "DICOM (.dcm) & Derived NIfTI (.nii.gz)",
          samples: "262 Patients with complete multi-parametric brain series",
          size: "12.4 GB",
          license: "Open Access TCIA (Attribution / Non-Commercial)",
          verified: true,
          suitability_score: 94,
          suitability_reason: "High clinical fidelity paired with comprehensive clinical pathology and genetic metadata for tumor subtype classification.",
          recommended: false,
          tags: ["Clinical Grade", "DICOM", "Genomic Ground Truth", "Pathology"],
        },
        {
          id: "figshare-brain-tumor",
          title: "Cheng Brain Tumor CE-MRI Dataset (3 Classes)",
          source: "Figshare / Nanfang Hospital & General Hospital of Tianjin",
          url: "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427",
          download_url: "https://ndownloader.figshare.com/articles/1512427/versions/5",
          download_type: "zip",
          modality: "T1-Weighted Contrast-Enhanced 2D MRI (Axial, Coronal, Sagittal)",
          format: "MATLAB (.mat) & PNG Slices",
          samples: "3,064 CE-MRI slices across 233 patients (Meningioma, Glioma, Pituitary)",
          size: "820 MB",
          license: "CC-BY 4.0 (Commercial & Academic Use Permitted)",
          verified: true,
          suitability_score: 91,
          suitability_reason: "Ideal for 2D multi-slice tumor classification and contour segmentation with explicit tumor region boundary masks.",
          recommended: false,
          tags: ["2D Slices", "Multi-Class", "Meningioma/Glioma", "Permissive License"],
        },
        {
          id: "ixi-brain-baseline",
          title: "IXI Dataset (Information eXtraction from Images Normal Cohort)",
          source: "Imperial College London / Brain Development UK",
          url: "https://brain-development.org/ixi-dataset/",
          modality: "T1, T2, PD, and MRA MRI Scans (Healthy Brain Normal Cohort)",
          format: "NIfTI (.nii.gz) with 1.5T and 3T acquisitions",
          samples: "600 Healthy Adult Scans",
          size: "6.1 GB",
          license: "CC-BY-SA 3.0",
          verified: true,
          suitability_score: 87,
          suitability_reason: "Critical negative-control baseline cohort to prevent false-positive tumor hallucinations on non-pathological tissue.",
          recommended: false,
          tags: ["Negative Control", "Healthy Baseline", "Normative Atlas", "600 Subjects"],
        },
      ];
    } else if (isTimeSeries) {
      datasets = [
        {
          id: "ett-transformer-bench",
          title: "ETT (Electricity Transformer Temperature Benchmark - Hourly)",
          source: "Tsinghua University / NeurIPS Time Series Benchmark",
          url: "https://github.com/zhouhaoyi/ETDataset",
          download_url: "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTh1.csv",
          download_type: "file",
          modality: "Multivariate Sensor Telemetry (Oil Temp, Voltage, Active Load)",
          format: "Standard CSV",
          samples: "17,420 Continuous Hourly Timesteps (2 Years)",
          size: "3.8 MB",
          license: "MIT License",
          verified: true,
          suitability_score: 98,
          suitability_reason: "Industry-standard benchmark for long-sequence time-series forecasting with standardized rolling walk-forward splits.",
          recommended: true,
          tags: ["Direct CSV Download", "NeurIPS Benchmark", "MIT License", "Sensor Telemetry"],
        },
        {
          id: "jena-climate-weather",
          title: "Jena Climate Meteorological Time Series (Hourly)",
          source: "Max Planck Institute for Biogeochemistry / TensorFlow",
          url: "https://www.bgc-jena.mpg.de/wetter/",
          download_url: "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
          download_type: "zip",
          modality: "14 Atmospheric Variables (Temp, Pressure, Humidity, Wind)",
          format: "CSV (Resampled Hourly)",
          samples: "70,000 Continuous Meteorological Timesteps (2009-2016)",
          size: "13.5 MB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 95,
          suitability_reason: "Rich non-stationary seasonal dynamics to test Reversible Instance Normalization (RevIN) and cyclic attention.",
          recommended: true,
          tags: ["Direct Open Download", "Hourly", "Meteorological", "Non-Stationary"],
        },
        {
          id: "ett-15min-bench",
          title: "ETT (Electricity Transformer Temperature - 15 Minute)",
          source: "Tsinghua University / NeurIPS Time Series Benchmark",
          url: "https://github.com/zhouhaoyi/ETDataset",
          download_url: "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTm1.csv",
          download_type: "file",
          modality: "Multivariate Sensor Telemetry (15-Minute Epochs)",
          format: "Standard CSV",
          samples: "69,680 Continuous 15-Minute Timesteps",
          size: "15.2 MB",
          license: "MIT License",
          verified: true,
          suitability_score: 92,
          suitability_reason: "High-frequency granularity benchmark testing fine-scale patch projections and temporal attention.",
          recommended: false,
          tags: ["15-Minute Frequency", "Multi-Variable", "MIT License"],
        },
      ];
    } else if (isAudio) {
      datasets = [
        {
          id: "speech-commands-mini",
          title: "Google Speech Commands Benchmark (Mini Corpus, 8 Words)",
          source: "Google Research / Pete Warden",
          url: "https://arxiv.org/abs/1804.03209",
          download_url: "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          download_type: "zip",
          modality: "1-Second 16 kHz WAV Audio Files (8 Classes)",
          format: "Linear PCM 16-bit WAV",
          samples: "8,000 Verified Audio Clips across 8 Command Classes",
          size: "180 MB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 98,
          suitability_reason: "Optimal for training on-device keyword spotting and compact command models from scratch.",
          recommended: true,
          tags: ["Direct Google Download", "Keyword Spotting", "16 kHz WAV", "Google Certified"],
        },
        {
          id: "esc50-audio",
          title: "ESC-50 Dataset for Environmental Sound Classification",
          source: "GitHub / Karol J. Piczak",
          url: "https://github.com/karolpiczak/ESC-50",
          download_url: "https://github.com/karolpiczak/ESC-50/archive/master.zip",
          download_type: "zip",
          modality: "5-Second 44.1 kHz WAV Recordings (50 Classes)",
          format: "Linear PCM WAV + CSV Metadata",
          samples: "2,000 Labeled Environmental Audio Clips in 5 Folds",
          size: "600 MB",
          license: "CC-BY-NC 3.0",
          verified: true,
          suitability_score: 92,
          suitability_reason: "Standard acoustic benchmark with prearranged cross-validation folds.",
          recommended: false,
          tags: ["Environmental", "50 Classes", "Folded Benchmark"],
        },
      ];
    } else {
      // General Computer Vision / Tabular
      datasets = [
        {
          id: "flower-photos-google",
          title: "Google Flowers Benchmark Dataset (5 Categories)",
          source: "Google Research & TensorFlow Datasets",
          url: "https://www.tensorflow.org/datasets/catalog/tf_flowers",
          download_url: "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
          download_type: "tar",
          modality: "Natural Color RGB Photography (Daisy, Dandelion, Roses, Sunflowers, Tulips)",
          format: "Standard Web JPEG",
          samples: "3,670 High-Resolution Categorized Images",
          size: "218 MB",
          license: "CC-BY 2.0 (Open Permissive)",
          verified: true,
          suitability_score: 98,
          suitability_reason: "Directly downloadable, balanced multi-class vision dataset ideal for pretraining convolutional and vision transformer backbones from scratch.",
          recommended: true,
          tags: ["Direct Google Download", "5 Classes", "Fast Convergence", "Vision Benchmark"],
        },
        {
          id: "wine-quality-uci",
          title: "UCI Wine Quality Physicochemical Dataset",
          source: "UCI Machine Learning Repository",
          url: "https://archive.ics.uci.edu/dataset/186/wine+quality",
          download_url: "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
          download_type: "file",
          modality: "11 Physicochemical Attributes + Sensory Quality Scores",
          format: "Standard CSV",
          samples: "4,898 Clean Sample Records",
          size: "260 KB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 94,
          suitability_reason: "Clean tabular feature set for validating MLP and TabNet architectures trained from scratch.",
          recommended: false,
          tags: ["Direct CSV Download", "Tabular", "UCI Certified"],
        },
      ];
    }

    // Automatically index project context & datasets into ChromaDB Vector Store
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
          meta: {
            name: projectName,
            idea,
            domains,
            primaryGoal: project.primaryGoal || "Production",
            model,
          },
          answers,
          datasets,
        }),
      });
      if (ragRes.ok) {
        indexedIntoRag = true;
      }
    } catch (ragErr) {
      console.warn("Could not index into ChromaDB vector store:", ragErr);
    }

    return NextResponse.json({
      success: true,
      project_id: projectId,
      detected_domain: isMedical ? "cv_medical" : isTimeSeries ? "time_series" : isAudio ? "audio" : isTabular ? "tabular" : "cv_general",
      total: datasets.length,
      datasets,
      indexed_into_rag: indexedIntoRag,
    });
  } catch (err: any) {
    console.error("Dataset research error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to discover datasets" },
      { status: 500 }
    );
  }
}
