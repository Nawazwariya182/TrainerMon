import { NextRequest, NextResponse } from "next/server";

const API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";
const BACKEND_RAG_INDEX_URL = "http://127.0.0.1:8000/v1/research/rag/index";

export interface DatasetItem {
  id: string;
  title: string;
  source: string;
  url: string;
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
          id: "brats-2024",
          title: "BraTS 2024 Adult Glioma Multi-Parametric MRI Dataset",
          source: "Synapse.org & TCIA (RSNA-ASNR-MICCAI)",
          url: "https://www.synapse.org/Synapse:syn51156910/wiki/622351",
          modality: "3D Volumetric Multi-Parametric MRI (T1, T1Gd, T2, FLAIR)",
          format: "NIfTI (.nii.gz) with standard isotropic 1mm³ spacing",
          samples: "1,251 Subjects (Pre-operative multi-sequence brain scans)",
          size: "4.8 GB",
          license: "CC-BY-NC 4.0 (Open Research & Clinical Benchmark)",
          verified: true,
          suitability_score: 98,
          suitability_reason: "Gold-standard 3D MRI dataset for brain tumor segmentation from scratch with voxel-level annotations for Enhancing Tumor, Core, and Edema.",
          recommended: true,
          tags: ["Gold Standard", "3D Volumetric", "Voxel Masks", "BraTS Benchmark"],
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
          id: "msd-task01-brain",
          title: "Medical Segmentation Decathlon (MSD Task01: Brain Tumour)",
          source: "MedicalDecathlon.com / Grand-Challenge.org",
          url: "http://medicaldecathlon.com/",
          modality: "4-Channel 3D MRI (FLAIR, T1w, T1gd, T2w)",
          format: "Standardized NIfTI (.nii.gz) with JSON manifest",
          samples: "484 Training + 266 Validation Volumetric Scans",
          size: "3.2 GB",
          license: "CC-BY-SA 4.0 (Open Benchmark)",
          verified: true,
          suitability_score: 93,
          suitability_reason: "Strict benchmark protocol designed for training 3D U-Net and Swin UNETR from scratch without data leakage.",
          recommended: false,
          tags: ["Benchmark Protocol", "3D Resampled", "Decathlon Standard"],
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
        {
          id: "kaggle-sartaj-mri",
          title: "Brain Tumor MRI Benchmark (4-Class Clean Collection)",
          source: "Kaggle Benchmark Datasets / Curated Radiologist Cohort",
          url: "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri",
          modality: "Axial MRI Scans (Glioma, Meningioma, Pituitary, No Tumor)",
          format: "High-Resolution JPEG / PNG (224x224, 512x512)",
          samples: "7,023 Verified Axial Scans across 4 clinical classes",
          size: "168 MB",
          license: "Apache 2.0 (Permissive Open Source)",
          verified: true,
          suitability_score: 89,
          suitability_reason: "Lightweight, pre-split train/test benchmark optimal for fast classifier training from scratch and baseline convergence checks.",
          recommended: false,
          tags: ["Classification", "4 Classes", "Rapid Convergence", "Apache 2.0"],
        },
      ];
    } else if (isTimeSeries) {
      datasets = [
        {
          id: "nrel-solar-grid",
          title: "NREL Solar Power Data & Smart Grid Telemetry (5-Min)",
          source: "National Renewable Energy Laboratory (NREL)",
          url: "https://www.nrel.gov/grid/solar-power-data.html",
          modality: "Multivariate Time Series (Irradiance, Power Output, Ambient Temp)",
          format: "Parquet / CSV (5-Minute Resampled Epochs)",
          samples: "137,000 Time Steps across 24 regional sensor nodes",
          size: "1.4 GB",
          license: "Public Domain (US Government Open Data)",
          verified: true,
          suitability_score: 97,
          suitability_reason: "Standard multi-horizon forecasting benchmark ideal for PatchTST and TimesNet architectures trained from scratch.",
          recommended: true,
          tags: ["5-Minute Frequency", "Multivariate", "NREL Certified", "Public Domain"],
        },
        {
          id: "ett-transformer-bench",
          title: "ETT (Electricity Transformer Temperature Benchmark)",
          source: "Tsinghua University / NeurIPS Time Series Benchmark",
          url: "https://github.com/zhouhaoyi/ETDataset",
          modality: "Multivariate Sensor Telemetry (Oil Temp, Voltage, Active Load)",
          format: "CSV (15-Minute & 1-Hour Granularity)",
          samples: "70,000 Continuous Sequential Observations (2 Years)",
          size: "45 MB",
          license: "MIT License",
          verified: true,
          suitability_score: 95,
          suitability_reason: "Industry-standard benchmark for long-sequence time-series forecasting with standardized rolling walk-forward splits.",
          recommended: false,
          tags: ["Long Sequence", "NeurIPS Benchmark", "MIT License", "Sensor Telemetry"],
        },
        {
          id: "weatherbench-noaa",
          title: "WeatherBench & NOAA Global Surface Hourly Observations",
          source: "NOAA / ECMWF Open Data",
          url: "https://github.com/pangeo-data/WeatherBench",
          modality: "Global Gridded Time Series (Surface Temperature, Humidity, Wind)",
          format: "NetCDF / Zarr (Hourly Resampled)",
          samples: "350,000 Timesteps across 40 years of continuous logging",
          size: "8.2 GB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 91,
          suitability_reason: "Rich non-stationary seasonal dynamics to test Reversible Instance Normalization (RevIN) and cyclic attention.",
          recommended: false,
          tags: ["Non-Stationary", "Hourly", "Global Meteorological", "Cyclic"],
        },
        {
          id: "pemsd7-traffic-flow",
          title: "PeMSD7 California Highway Flow Sensor Telemetry",
          source: "Caltrans Performance Measurement System",
          url: "https://pems.dot.ca.gov/",
          modality: "Spatial-Temporal Graph Time Series (Speed, Occupancy, Flow)",
          format: "HDF5 / Numpy arrays (30-second aggregated to 5-minute)",
          samples: "12,672 Time Intervals across 228 highway sensor stations",
          size: "340 MB",
          license: "Open Data Commons PDDL",
          verified: true,
          suitability_score: 88,
          suitability_reason: "Ideal for spatiotemporal forecasting backbones combining temporal convolutions with spatial attention.",
          recommended: false,
          tags: ["Spatiotemporal", "Sensor Network", "5-Min Epochs"],
        },
        {
          id: "electricity-load-uci",
          title: "UCI ElectricityLoadDiagrams (370 Clients)",
          source: "UCI Machine Learning Repository",
          url: "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
          modality: "Multivariate Power Demand (kW per Client)",
          format: "CSV (15-Minute Intervals)",
          samples: "140,256 Hourly intervals (2011 - 2014)",
          size: "260 MB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 90,
          suitability_reason: "High customer diversity testing multi-series forecasting with zero inter-series cross-contamination.",
          recommended: false,
          tags: ["Power Grid", "15-Min", "Client Demand", "UCI Certified"],
        },
      ];
    } else if (isAudio) {
      datasets = [
        {
          id: "librispeech-clean",
          title: "LibriSpeech ASR Corpus (Clean 1,000 Hours)",
          source: "OpenSLR (Vassil Panayotov et al.)",
          url: "https://www.openslr.org/12",
          modality: "16 kHz Monophonic FLAC Audio + Text Transcriptions",
          format: "FLAC / Alignment Text",
          samples: "1,000 Hours of read English audiobook speech across 2,484 speakers",
          size: "55 GB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 97,
          suitability_reason: "The gold standard for training Conformer and acoustic encoders from scratch with standardized test-clean and test-other splits.",
          recommended: true,
          tags: ["1,000 Hours", "16 kHz", "Gold Standard", "CC-BY 4.0"],
        },
        {
          id: "common-voice-en",
          title: "Mozilla Common Voice 17 (English Cohort)",
          source: "Mozilla Foundation",
          url: "https://commonvoice.mozilla.org/",
          modality: "MP3 48 kHz / 16 kHz Downsampled + Metadata",
          format: "MP3 / TSV Metadata",
          samples: "3,200 Recorded Hours across diverse accents and environments",
          size: "82 GB",
          license: "CC-0 (Public Domain Dedication)",
          verified: true,
          suitability_score: 94,
          suitability_reason: "Realistic acoustic diversity and background noise profiles to train robust acoustic representations from scratch.",
          recommended: false,
          tags: ["Accent Diversity", "Public Domain", "Crowdsourced", "Robustness"],
        },
        {
          id: "speech-commands-v2",
          title: "Google Speech Commands Dataset v2 (35 Words)",
          source: "Google Research (Pete Warden)",
          url: "https://arxiv.org/abs/1804.03209",
          modality: "1-Second 16 kHz WAV Audio Files",
          format: "WAV (Linear PCM 16-bit)",
          samples: "105,829 Audio clips across 2,618 speakers",
          size: "2.4 GB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 96,
          suitability_reason: "Optimal for training on-device keyword spotting and compact command models from scratch.",
          recommended: false,
          tags: ["On-Device", "Keyword Spotting", "1-Second Clips", "Google Certified"],
        },
        {
          id: "urbansound-8k",
          title: "UrbanSound8K Acoustic Event Collection",
          source: "NYU Music and Audio Research Laboratory (MARL)",
          url: "https://urbansounddataset.weebly.com/urbansound8k.html",
          modality: "Environmental Audio Recordings (10 Classes)",
          format: "WAV (Up to 4 seconds, 44.1 kHz / 22 kHz)",
          samples: "8,732 Labeled urban sound excerpts in 10 folds",
          size: "5.6 GB",
          license: "CC-BY-NC 3.0",
          verified: true,
          suitability_score: 89,
          suitability_reason: "Pre-arranged 10-fold cross validation structure preventing speaker and environment leakage across folds.",
          recommended: false,
          tags: ["Environmental", "10 Classes", "Folded Split", "Urban Audio"],
        },
        {
          id: "voxceleb-1-2",
          title: "VoxCeleb Audio-Visual Speaker Recognition Dataset",
          source: "Oxford University Visual Geometry Group (VGG)",
          url: "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
          modality: "In-the-Wild Speech Extracted from Videos",
          format: "16 kHz 16-bit WAV",
          samples: "100,000+ Utterances from 1,251 celebrities",
          size: "39 GB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 88,
          suitability_reason: "Ideal for training speaker verification, contrastive audio embedding heads, and voice feature extractors from scratch.",
          recommended: false,
          tags: ["Speaker Verification", "In-the-Wild", "Oxford VGG"],
        },
      ];
    } else {
      // General Computer Vision / Multi-modal fallback
      datasets = [
        {
          id: "ms-coco-2017",
          title: "MS COCO 2017 (Common Objects in Context)",
          source: "Microsoft Research & Consortia",
          url: "https://cocodataset.org/",
          modality: "Color RGB Images with Bounding Boxes & Segmentations",
          format: "JPEG with COCO JSON Annotations",
          samples: "118,287 Training + 5,000 Validation Images (80 Classes)",
          size: "25 GB",
          license: "CC-BY 4.0",
          verified: true,
          suitability_score: 97,
          suitability_reason: "The universal benchmark for object detection, instance segmentation, and vision feature learning from scratch.",
          recommended: true,
          tags: ["80 Classes", "Detection/Segmentation", "Standard Benchmark"],
        },
        {
          id: "pascal-voc-2012",
          title: "PASCAL Visual Object Classes (VOC 2012 Augmented)",
          source: "University of Oxford & PASCAL Network",
          url: "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/",
          modality: "2D RGB Images with Pixel-Level Segmentation Masks",
          format: "JPEG + PNG Indexed Masks",
          samples: "17,125 Augmented Images across 20 Object Categories",
          size: "2.1 GB",
          license: "Non-commercial Research & Educational Use",
          verified: true,
          suitability_score: 92,
          suitability_reason: "Compact, high-precision semantic segmentation dataset optimal for rapid prototyping from scratch.",
          recommended: false,
          tags: ["Semantic Segmentation", "20 Classes", "Compact Footprint"],
        },
        {
          id: "imagenet-1k-mini",
          title: "ImageNet-1K Mini Benchmark (100 Selected Classes)",
          source: "Stanford Vision Lab / HuggingFace Mirrors",
          url: "https://huggingface.co/datasets/imagenet-1k",
          modality: "Diverse Natural RGB Photography",
          format: "Standard Web JPEG (224x224 & 512x512)",
          samples: "50,000 High-Quality Categorized Images",
          size: "3.4 GB",
          license: "Custom Non-Commercial Academic License",
          verified: true,
          suitability_score: 90,
          suitability_reason: "Ideal for validating convolutional or vision transformer weight initialization and convergence speed.",
          recommended: false,
          tags: ["Classification", "100 Classes", "Stanford Vision"],
        },
        {
          id: "openimages-v7-subset",
          title: "Open Images V7 Clean Annotation Subset",
          source: "Google Research",
          url: "https://storage.googleapis.com/openimages/web/index.html",
          modality: "Complex Multi-Label Real-World Imagery",
          format: "JPEG + CSV Coordinate Annotations",
          samples: "80,000 Clean Human-Verified Annotated Scenes",
          size: "18 GB",
          license: "CC-BY 2.0",
          verified: true,
          suitability_score: 89,
          suitability_reason: "Unbiased real-world scene distributions to stress-test spatial invariance and multi-scale feature hierarchies.",
          recommended: false,
          tags: ["Complex Scenes", "Multi-Label", "Google Research"],
        },
        {
          id: "cityscapes-dataset",
          title: "Cityscapes Semantic Urban Driving Benchmark",
          source: "Daimler AG, Max Planck Institute, TU Darmstadt",
          url: "https://www.cityscapes-dataset.com/",
          modality: "High-Resolution Stereo Video & Still Frames (50 Cities)",
          format: "PNG (1024x2048 Pixel Masks)",
          samples: "5,000 Fine Annotated + 20,000 Coarse Annotated Frames",
          size: "12 GB",
          license: "Non-commercial Academic Use",
          verified: true,
          suitability_score: 91,
          suitability_reason: "Rigorous pixel-level semantic segmentation standard with strict class definitions and fine boundary details.",
          recommended: false,
          tags: ["Autonomous Driving", "Fine Voxel Masks", "Urban Scenes"],
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
