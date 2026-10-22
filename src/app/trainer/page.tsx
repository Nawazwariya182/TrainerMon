"use client";

import React, { useState, useEffect, useRef, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  Sparkles,
  Send,
  ChevronDown,
  ChevronRight,
  Plus,
  Bot,
  User,
  CheckCircle2,
  Clock,
  Layers,
  Cpu,
  BrainCircuit,
  Eye,
  Info,
  Check,
  RotateCcw,
  Zap,
  HelpCircle,
  Lightbulb,
  ExternalLink,
  Sliders,
  AlertTriangle,
  Database,
  FileCode,
  Globe,
  BookmarkCheck,
  Edit3,
  Save,
  CheckSquare,
  Square,
  Trash2,
  Share2,
  Terminal,
  Copy,
  Download,
  FileText,
  FolderOpen,
  Play,
  CheckCheck,
  Code,
  Folder,
  GitBranch,
  RefreshCw,
} from "lucide-react";

interface Question {
  id: number;
  category: string;
  question: string;
  context: string;
  options: string[];
  ai_recommendation?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  timestamp: string;
}

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

type StudioStage = "questionnaire" | "datasets" | "dataset_preview" | "tech_research" | "code_studio";

const AVAILABLE_MODELS = [
  { id: "qwen3-8b", name: "Qwen3-8B", label: "Qwen3 8B (Recommended)", icon: BrainCircuit },
  { id: "qwen3.5-9b", name: "Qwen3.5-9B", label: "Qwen3.5 9B (Extended Context)", icon: Sparkles },
  { id: "qwen3-vl-8b", name: "Qwen3-VL-8B", label: "Qwen3-VL 8B (Multimodal)", icon: Eye },
];

function TrainerStudioContent() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get("projectId");
  const modelQuery = searchParams.get("model");

  // Project state
  const [project, setProject] = useState<any>(null);
  const [loadingProject, setLoadingProject] = useState<boolean>(true);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState<string>(modelQuery || "qwen3-8b");
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState<boolean>(false);

  // Studio Flow State
  const [studioStage, setStudioStage] = useState<StudioStage>("questionnaire");

  // Questionnaire state
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [answers, setAnswers] = useState<Record<number, { option?: string; custom?: string }>>({});
  const [complexity, setComplexity] = useState<string>("Moderate");
  const [loadingQuestions, setLoadingQuestions] = useState<boolean>(true);
  const [isFeasible, setIsFeasible] = useState<boolean>(true);
  const [sorryMessage, setSorryMessage] = useState<string | null>(null);
  const [isThinkingOpen, setIsThinkingOpen] = useState<boolean>(true);
  const [initialThinkingTrace, setInitialThinkingTrace] = useState<string>("");

  // Datasets state
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState<boolean>(false);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const [activePreviewDataset, setActivePreviewDataset] = useState<DatasetItem | null>(null);

  // Technical research state
  const [techResearch, setTechResearch] = useState<TechResearchPlan | null>(null);
  const [loadingTechResearch, setLoadingTechResearch] = useState<boolean>(false);
  const [isEditingTechResearch, setIsEditingTechResearch] = useState<boolean>(false);
  const [isResearchSubmitted, setIsResearchSubmitted] = useState<boolean>(false);

  // Code generation & workspace state
  const [codeFiles, setCodeFiles] = useState<Record<string, string>>({});
  const [loadingCode, setLoadingCode] = useState<boolean>(false);
  const [activeCodeFile, setActiveCodeFile] = useState<string>("train.py");
  const [workspacePath, setWorkspacePath] = useState<string>("");
  const [relativeWorkspacePath, setRelativeWorkspacePath] = useState<string>("");
  const [copiedFile, setCopiedFile] = useState<boolean>(false);
  const [showRunGuideModal, setShowRunGuideModal] = useState<boolean>(false);
  const [isIndexedInRag, setIsIndexedInRag] = useState<boolean>(false);

  // GitHub & Execution Status State
  const [showGitHubModal, setShowGitHubModal] = useState<boolean>(false);
  const [githubRepoUrl, setGithubRepoUrl] = useState<string>("");
  const [githubToken, setGithubToken] = useState<string>("");
  const [githubBranch, setGithubBranch] = useState<string>("main");
  const [githubCommitMsg, setGithubCommitMsg] = useState<string>("");
  const [isPushingGit, setIsPushingGit] = useState<boolean>(false);
  const [gitPushLogs, setGitPushLogs] = useState<string[]>([]);
  const [gitPushResult, setGitPushResult] = useState<{ success: boolean; message: string } | null>(null);
  const [executionStatus, setExecutionStatus] = useState<any>(null);
  const [isRefreshingStatus, setIsRefreshingStatus] = useState<boolean>(false);

  // Chat interface state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>("");
  const [isSendingChat, setIsSendingChat] = useState<boolean>(false);
  const [submitNotice, setSubmitNotice] = useState<string | null>(null);

  const chatScrollRef = useRef<HTMLDivElement>(null);

  // 1. Fetch project if projectId is supplied
  useEffect(() => {
    async function fetchProjectData() {
      if (!projectId) {
        setLoadingProject(false);
        return;
      }
      try {
        const res = await fetch("/api/projects");
        const data = await res.json();
        if (data.success && Array.isArray(data.projects)) {
          const matched = data.projects.find((p: any) => p.id === projectId);
          if (matched) {
            setProject(matched);
            if (matched.model) {
              setSelectedModel(matched.model);
            }
          }
        }
      } catch (err) {
        console.error("Error fetching project:", err);
      } finally {
        setLoadingProject(false);
      }
    }
    fetchProjectData();
  }, [projectId]);

  // 2. Generate questionnaire when project or model is ready
  useEffect(() => {
    async function generateQuestionnaire() {
      setLoadingQuestions(true);
      try {
        const res = await fetch("/api/questionnaire/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: project?.name || "Machine Learning Project",
            idea: project?.idea || project?.description || "Train an optimized domain-specific model.",
            domains: project?.domains || ["Image / CV"],
            learningApproaches: project?.learningApproaches || ["Supervised Learning", "Deep Learning"],
            primaryGoal: project?.primaryGoal || "Production application",
            model: selectedModel,
          }),
        });

        const data = await res.json();

        // Handle Infeasible / Impossible Request
        if (data.success && data.is_feasible === false) {
          setIsFeasible(false);
          setSorryMessage(
            data.sorry_message ||
              "Sorry, your project request cannot be trained from scratch as specified. Please review architectural limits, scaling laws, or dataset constraints."
          );
          setQuestions([]);
          setInitialThinkingTrace(data.thinking_trace || "");
          setChatMessages([
            {
              id: "infeasible-init",
              role: "assistant",
              content:
                data.sorry_message ||
                "Sorry, your project request cannot be trained from scratch as specified. I am here to help you reframe the architecture and compute parameters into an achievable pretraining plan.",
              thinking: data.thinking_trace || undefined,
              timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            },
          ]);
          return;
        }

        if (data.success && Array.isArray(data.questions) && data.questions.length > 0) {
          setIsFeasible(true);
          setSorryMessage(null);
          setQuestions(data.questions);
          setComplexity(data.complexity || "Moderate");
          setInitialThinkingTrace(data.thinking_trace || "");

          const domainTitle = data.detected_domain === "cv_medical"
            ? "Computer Vision (Medical Scans & Volumetric Tensors)"
            : data.detected_domain === "cv_general"
            ? "Computer Vision (2D/3D Image & Video)"
            : data.detected_domain === "audio"
            ? "Audio & Speech Processing"
            : data.detected_domain === "time_series"
            ? "Time Series & Sensor Telemetry"
            : data.detected_domain === "tabular"
            ? "Tabular & Structured Learning"
            : "Foundation Machine Learning";

          // Initial assistant message in chat
          setChatMessages([
            {
              id: "init-1",
              role: "assistant",
              content: `Hello! I have evaluated your from-scratch model training project **"${project?.name || "ML Project"}"** (${domainTitle}) as **${data.complexity || "Moderate"} Complexity**.

I have structured a comprehensive **${data.questions.length}-question architectural questionnaire** tailored specifically to training this model from scratch (covering input tensor formatting, neural backbones, domain preprocessing, loss objectives, weight initialization, augmentations, and evaluation benchmarks).

You can answer each question in the main panel, and ask me anytime here in chat if you want advice on trade-offs!`,
              thinking: data.thinking_trace || undefined,
              timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            },
          ]);
        }
      } catch (err) {
        console.error("Failed to generate questionnaire:", err);
      } finally {
        setLoadingQuestions(false);
      }
    }

    if (!loadingProject) {
      generateQuestionnaire();
    }
  }, [loadingProject, project, selectedModel]);

  // Scroll chat to bottom on new messages
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, isSendingChat]);

  // Handle Chat Submit
  const handleSendChat = async (e?: React.FormEvent, customPrompt?: string) => {
    if (e) e.preventDefault();
    const promptToSend = customPrompt || chatInput.trim();
    if (!promptToSend || isSendingChat) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: promptToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    if (!customPrompt) setChatInput("");
    setIsSendingChat(true);

    try {
      const res = await fetch("/api/studio/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...chatMessages, userMsg].map((m) => ({
            role: m.role,
            content: m.content,
          })),
          model: selectedModel,
          projectContext: {
            project,
            currentQuestion: questions[currentQuestionIndex],
            answers,
            datasets: datasets.filter((d) => selectedDatasetIds.includes(d.id)),
            techResearch,
          },
        }),
      });

      const data = await res.json();
      if (data.success && data.message) {
        setChatMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.message.content,
            thinking: data.message.thinking || undefined,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
      } else {
        throw new Error(data.error || "Chat failed");
      }
    } catch (err: any) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `⚠️ Error generating response: ${err.message}. Please check inference engine status.`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsSendingChat(false);
    }
  };

  // Questionnaire Answer Handlers
  const handleSelectOption = (option: string) => {
    setAnswers((prev) => ({
      ...prev,
      [currentQuestionIndex]: {
        ...prev[currentQuestionIndex],
        option,
        question: questions[currentQuestionIndex]?.question,
        category: questions[currentQuestionIndex]?.category,
      },
    }));
  };

  const handleCustomTextChange = (custom: string) => {
    setAnswers((prev) => ({
      ...prev,
      [currentQuestionIndex]: {
        ...prev[currentQuestionIndex],
        custom,
        question: questions[currentQuestionIndex]?.question,
        category: questions[currentQuestionIndex]?.category,
      },
    }));
  };

  const currentQ = questions[currentQuestionIndex];
  const currentAnswer = answers[currentQuestionIndex] || {};

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((prev) => prev - 1);
    }
  };

  const handleSkipOrAiDefault = () => {
    if (currentQ?.ai_recommendation) {
      handleSelectOption(currentQ.ai_recommendation);
    }
    handleNext();
  };

  // Submit Questionnaire -> Trigger AI Research & Dataset Discovery
  const handleSubmitQuestionnaire = async () => {
    setStudioStage("datasets");
    setLoadingDatasets(true);
    setSubmitNotice("Requirements captured! The AI Agentic Orchestrator is now searching and verifying suitable datasets...");

    // Announce in chat
    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `**Orchestrator Stage Activated: Dataset Discovery & Source Verification**

I have received your ${questions.length} architectural choices. I am now querying scientific repositories (Synapse/TCIA, HuggingFace, Kaggle, OpenSLR, NREL) to find 5–7 verified datasets matching your exact modality and licensing requirements.`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);

    try {
      const res = await fetch("/api/research/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: {
            id: projectId,
            name: project?.name,
            idea: project?.idea || project?.description,
            domains: project?.domains,
            learningApproaches: project?.learningApproaches,
            primaryGoal: project?.primaryGoal,
            model: selectedModel,
          },
          answers,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      if (data.success && Array.isArray(data.datasets) && data.datasets.length > 0) {
        setDatasets(data.datasets);
        // Default select recommended or first dataset
        const rec = data.datasets.find((d: DatasetItem) => d.recommended) || data.datasets[0];
        if (rec) {
          setSelectedDatasetIds([rec.id]);
        }
        setSubmitNotice(null);

        setChatMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `I have discovered **${data.datasets.length} verified datasets** for your project.

Recommended primary dataset: **${rec.title}** (${rec.license}, ${rec.samples}).
You can review licenses, click **View Source** to inspect any dataset website in the main panel, select which ones to train with, or click **Send to Assistant** to ask me anything about a dataset!`,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
      } else {
        throw new Error(data.error || "No datasets discovered");
      }
    } catch (err: any) {
      console.error("Error fetching datasets:", err);
      setSubmitNotice(`Dataset research encountered an issue: ${err.message}. Retrying...`);
    } finally {
      setLoadingDatasets(false);
    }
  };

  // Toggle Dataset Selection
  const handleToggleDataset = (dsId: string) => {
    setSelectedDatasetIds((prev) =>
      prev.includes(dsId) ? prev.filter((id) => id !== dsId) : [...prev, dsId]
    );
  };

  // Send Dataset to 20% Assistant Dock
  const handleSendDatasetToChat = (ds: DatasetItem) => {
    const prompt = `Can you analyze the "${ds.title}" dataset from ${ds.source}? Specifically, what are its preprocessing requirements for training our model from scratch, and how do its annotations (${ds.samples}, ${ds.format}) align with our loss objective?`;
    handleSendChat(undefined, prompt);
  };

  // Open In-Panel Dataset Preview
  const handleViewDatasetSource = (ds: DatasetItem) => {
    setActivePreviewDataset(ds);
    setStudioStage("dataset_preview");
  };

  // Trigger Technical Research Blueprint
  const handleGenerateTechResearch = async () => {
    setStudioStage("tech_research");
    setLoadingTechResearch(true);

    const chosenDatasets = datasets.filter((d) => selectedDatasetIds.includes(d.id));

    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `**Orchestrator Stage Activated: Technical Architecture & Training Blueprint**

Synthesizing neural backbones, layer formulations, Soft Dice + Focal loss equations, generic GPU memory estimates, and implementation tasks for ${chosenDatasets.length > 0 ? chosenDatasets.map((d) => d.title).join(", ") : "your selected dataset"}...`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);

    try {
      const res = await fetch("/api/research/tech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: {
            id: projectId,
            name: project?.name,
            idea: project?.idea || project?.description,
            domains: project?.domains,
            primaryGoal: project?.primaryGoal,
            model: selectedModel,
          },
          selectedDatasets: chosenDatasets,
          answers,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      if (data.success && data.research) {
        setTechResearch(data.research);
        setChatMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `Technical Blueprint Generated!

**Primary Architecture**: ${data.research.architecture.selected_name}
**Estimated Compute**: ${data.research.hardware_and_compute.estimated_vram_footprint} (${data.research.hardware_and_compute.estimated_parameters})
**Loss Formulation**: ${data.research.loss_formulation.primary_loss}

You can directly edit any section of the research plan in the main panel, or ask me here to modify algorithms, hyperparameters, or TODO tasks!`,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
      } else {
        throw new Error(data.error || "Failed to generate technical blueprint");
      }
    } catch (err: any) {
      console.error("Error generating technical blueprint:", err);
    } finally {
      setLoadingTechResearch(false);
    }
  };

  // Toggle TODO status
  const handleToggleTodo = (todoId: number) => {
    if (!techResearch) return;
    setTechResearch({
      ...techResearch,
      implementation_todos: techResearch.implementation_todos.map((t) =>
        t.id === todoId
          ? { ...t, status: t.status === "completed" ? "pending" : "completed" }
          : t
      ),
    });
  };

  // Add custom TODO
  const handleAddTodo = () => {
    if (!techResearch) return;
    const newId = techResearch.implementation_todos.length + 1;
    setTechResearch({
      ...techResearch,
      implementation_todos: [
        ...techResearch.implementation_todos,
        {
          id: newId,
          title: "New Custom Engineering Step",
          description: "Specify implementation instructions or verification benchmarks.",
          status: "pending",
        },
      ],
    });
  };

  // Submit Research (Dummy for now as specified)
  const handleSubmitResearchPlan = () => {
    setIsResearchSubmitted(true);
    setSubmitNotice("Research Plan Finalized & Locked! All dataset selections, neural architectures, loss formulations, and implementation tasks have been saved into studio memory.");
    setTimeout(() => {
      setSubmitNotice(null);
    }, 7000);
  };

  // Generate & Save Production-Grade Code Pipeline
  const handleGenerateCodePipeline = async () => {
    setStudioStage("code_studio");
    setLoadingCode(true);

    const chosenDatasets = datasets.filter((d) => selectedDatasetIds.includes(d.id));

    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `**Orchestrator Stage Activated: Code Generation & Multi-Dataset Harmonization**

Synthesizing complete, runnable PyTorch code suite for **${project?.name || "Pretraining Project"}**.
Harmonizing ${chosenDatasets.length > 0 ? chosenDatasets.map((d) => d.title).join(", ") : "all selected datasets"} into standardized tensor geometry, writing data loaders, neural backbone, composite loss functions, AMP mixed precision, validation scoring, and saving all files directly into \`workspace_projects/${projectId || "project"}\`...`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);

    try {
      const res = await fetch("/api/research/code/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId: projectId || "proj_" + Math.random().toString(36).substring(2, 9),
          project: {
            id: projectId,
            name: project?.name,
            idea: project?.idea || project?.description,
            domains: project?.domains,
            learningApproaches: project?.learningApproaches,
            primaryGoal: project?.primaryGoal,
            model: selectedModel,
          },
          selectedDatasets: chosenDatasets,
          answers,
          techResearch,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      if (data.success && data.files) {
        setCodeFiles(data.files);
        setWorkspacePath(data.workspace_path || "");
        setRelativeWorkspacePath(data.relative_workspace_path || "");
        setIsIndexedInRag(!!data.indexed_into_rag);

        if (data.files["train.py"]) {
          setActiveCodeFile("train.py");
        } else {
          setActiveCodeFile(Object.keys(data.files)[0]);
        }

        if (data.files["EXECUTION_STATUS.json"]) {
          try {
            setExecutionStatus(JSON.parse(data.files["EXECUTION_STATUS.json"]));
          } catch {}
        }

        setChatMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `**Complete Code Pipeline Generated & Saved!** 🚀

- **Files Created**: ${Object.keys(data.files).length} files
- **Dedicated Folder**: \`${data.relative_workspace_path}\`
- **ChromaDB Vector Store**: Automatically indexed for instant Q&A.

You can inspect all scripts in the Code Studio, copy or download individual files, or ask me here to explain any function or adjust hyperparameters!`,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
      } else {
        throw new Error(data.error || "Failed to generate code files");
      }
    } catch (err: any) {
      console.error("Code generation error:", err);
      setChatMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `⚠️ Code generation error: ${err.message}. Please check inference server status.`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setLoadingCode(false);
    }
  };

  const handleCopyCode = (content: string) => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  const handleDownloadFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSendCodeFileToChat = (filename: string) => {
    const prompt = `Can you provide a walkthrough of ${filename}? Specifically explain its architecture, how multi-dataset tensors are processed, and the key functions to watch during pretraining.`;
    handleSendChat(undefined, prompt);
  };

  const fetchExecutionStatus = async () => {
    const targetId = projectId || project?.id;
    if (!targetId) return;
    setIsRefreshingStatus(true);
    try {
      const res = await fetch(`/api/workspace/status?projectId=${encodeURIComponent(targetId)}`);
      const data = await res.json();
      if (data.success && data.status) {
        setExecutionStatus(data.status);
      }
    } catch (err) {
      console.warn("Could not fetch execution status:", err);
    } finally {
      setIsRefreshingStatus(false);
    }
  };

  const handlePushToGitHub = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!githubRepoUrl.trim()) return;
    setIsPushingGit(true);
    setGitPushLogs(["Connecting to local workspace and Git engine..."]);
    setGitPushResult(null);

    const targetId = projectId || project?.id;

    try {
      const res = await fetch("/api/workspace/git", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          projectId: targetId,
          repoUrl: githubRepoUrl.trim(),
          branch: githubBranch.trim() || "main",
          token: githubToken.trim(),
          commitMessage:
            githubCommitMsg.trim() ||
            `Initial commit: ${project?.name || "ML-Train"} pretraining pipeline from scratch`,
        }),
      });

      const data = await res.json();
      if (data.logs && Array.isArray(data.logs)) {
        setGitPushLogs(data.logs);
      }

      if (data.success) {
        setGitPushResult({
          success: true,
          message: data.message || "Successfully pushed to GitHub!",
        });
        setChatMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `**GitHub Repository Push Completed!** 🚀

- **Remote URL**: \`${githubRepoUrl}\`
- **Branch**: \`${githubBranch || "main"}\`
- **Files Staged**: All workspace source code, manifests, and configs (excluding raw caches and binary weights).`,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
      } else {
        throw new Error(data.error || "Git push failed");
      }
    } catch (err: any) {
      setGitPushResult({
        success: false,
        message: err.message || "Push failed. Please check repository URL and permissions.",
      });
    } finally {
      setIsPushingGit(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-white text-slate-900 overflow-hidden font-sans">
      {/* Top Studio Header Bar */}
      <header className="h-14 border-b border-slate-200 bg-white/95 backdrop-blur-xs px-4 sm:px-6 flex items-center justify-between shrink-0 z-30">
        <div className="flex items-center space-x-3">
          <Link
            href="/"
            className="flex items-center space-x-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 px-2.5 py-1.5 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Projects</span>
          </Link>
          <span className="text-slate-300">/</span>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-sm text-slate-900">
              Trainer Studio
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium border border-slate-200">
              {project?.name || "Pretraining Project"}
            </span>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200/60">
              {studioStage === "questionnaire"
                ? "Phase 1: Architecture Questionnaire"
                : studioStage === "datasets" || studioStage === "dataset_preview"
                ? "Phase 2A: Dataset Discovery"
                : studioStage === "tech_research"
                ? "Phase 2B: Technical Blueprint"
                : "Phase 3: Code Studio & Pipeline"}
            </span>
          </div>
        </div>

        {/* Right Status Indicator */}
        <div className="flex items-center space-x-3">
          <div className="hidden sm:flex items-center space-x-1.5 text-xs text-slate-600 bg-slate-50 border border-slate-200 px-3 py-1 rounded-full">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium text-slate-700">Engine Live</span>
            <span className="text-slate-400">•</span>
            <span className="text-indigo-600 font-semibold">
              {AVAILABLE_MODELS.find((m) => m.id === selectedModel)?.name}
            </span>
          </div>
        </div>
      </header>

      {/* Main Studio Body: 20% Left Chat Dock + 80% Main Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* ======================================================== */}
        {/* LEFT DOCK (20%): Thinking & Conversational Assistant     */}
        {/* ======================================================== */}
        <aside className="w-[300px] xl:w-[340px] border-r border-slate-200 bg-white flex flex-col shrink-0 z-20">
          {/* Assistant Header & Orchestrator Stage Pill */}
          <div className="p-3.5 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-xs">
                <Bot className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-bold text-xs text-slate-900 leading-tight">
                  Studio Architect
                </h3>
                <p className="text-[10px] text-slate-500">
                  {studioStage === "questionnaire"
                    ? "Interactive Questioning"
                    : studioStage === "datasets"
                    ? "Dataset Verification"
                    : studioStage === "tech_research"
                    ? "Technical RAG Synthesis"
                    : "Code Studio & Pipeline"}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-1">
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-white border border-slate-200 font-medium text-slate-600">
                20% Split
              </span>
            </div>
          </div>

          {/* Orchestrator Progress Stepper */}
          <div className="px-2.5 py-2 bg-slate-100/70 border-b border-slate-200/80 flex items-center justify-between text-[10px] text-slate-600">
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  studioStage === "questionnaire" ? "bg-indigo-600 animate-pulse" : "bg-emerald-500"
                }`}
              />
              <span className={studioStage === "questionnaire" ? "font-bold text-indigo-700" : ""}>
                1. Q&A
              </span>
            </div>
            <ChevronRight className="w-2.5 h-2.5 text-slate-400" />
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  studioStage === "datasets" || studioStage === "dataset_preview"
                    ? "bg-indigo-600 animate-pulse"
                    : studioStage === "tech_research" || studioStage === "code_studio"
                    ? "bg-emerald-500"
                    : "bg-slate-300"
                }`}
              />
              <span
                className={
                  studioStage === "datasets" || studioStage === "dataset_preview"
                    ? "font-bold text-indigo-700"
                    : ""
                }
              >
                2. Data
              </span>
            </div>
            <ChevronRight className="w-2.5 h-2.5 text-slate-400" />
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  studioStage === "tech_research"
                    ? "bg-indigo-600 animate-pulse"
                    : studioStage === "code_studio"
                    ? "bg-emerald-500"
                    : "bg-slate-300"
                }`}
              />
              <span className={studioStage === "tech_research" ? "font-bold text-indigo-700" : ""}>
                3. Plan
              </span>
            </div>
            <ChevronRight className="w-2.5 h-2.5 text-slate-400" />
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  studioStage === "code_studio" ? "bg-indigo-600 animate-pulse" : "bg-slate-300"
                }`}
              />
              <span className={studioStage === "code_studio" ? "font-bold text-indigo-700" : ""}>
                4. Code
              </span>
            </div>
          </div>

          {/* Chat Messages List */}
          <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-3.5 space-y-3.5">
            {/* Thinking Drawer (if trace present) */}
            {initialThinkingTrace && (
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/50">
                <button
                  type="button"
                  onClick={() => setIsThinkingOpen((prev) => !prev)}
                  className="w-full px-3 py-1.5 flex items-center justify-between text-[11px] font-semibold text-slate-700 hover:bg-slate-100 transition-colors"
                >
                  <div className="flex items-center space-x-1.5 text-indigo-600">
                    <BrainCircuit className="w-3.5 h-3.5" />
                    <span>Architect Reasoning & Thinking</span>
                  </div>
                  {isThinkingOpen ? (
                    <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                  )}
                </button>
                {isThinkingOpen && (
                  <div className="p-2.5 text-[11px] font-mono text-slate-600 bg-white border-t border-slate-100 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                    {initialThinkingTrace}
                  </div>
                )}
              </div>
            )}

            {chatMessages.map((msg) => {
              const isAssistant = msg.role === "assistant";
              return (
                <div
                  key={msg.id}
                  className={`flex items-start space-x-2 ${
                    isAssistant ? "justify-start" : "justify-end"
                  }`}
                >
                  {isAssistant && (
                    <div className="w-6 h-6 rounded-md bg-indigo-50 border border-indigo-200 text-indigo-600 flex items-center justify-center shrink-0 mt-0.5">
                      <Sparkles className="w-3 h-3" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] rounded-2xl p-3 text-xs leading-relaxed shadow-2xs ${
                      isAssistant
                        ? "bg-slate-50 text-slate-800 border border-slate-200/80"
                        : "bg-indigo-600 text-white font-medium"
                    }`}
                  >
                    {msg.thinking && (
                      <details className="mb-2 text-[10px] text-slate-500 border-b border-slate-200 pb-1.5">
                        <summary className="cursor-pointer font-mono font-medium hover:text-indigo-600">
                          View Deep Reasoning
                        </summary>
                        <p className="mt-1 font-mono text-slate-600 whitespace-pre-wrap">
                          {msg.thinking}
                        </p>
                      </details>
                    )}
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    <div
                      className={`text-[9px] mt-1 text-right ${
                        isAssistant ? "text-slate-400" : "text-indigo-200"
                      }`}
                    >
                      {msg.timestamp}
                    </div>
                  </div>

                  {!isAssistant && (
                    <div className="w-6 h-6 rounded-md bg-slate-900 text-white flex items-center justify-center shrink-0 mt-0.5">
                      <User className="w-3 h-3" />
                    </div>
                  )}
                </div>
              );
            })}

            {isSendingChat && (
              <div className="flex items-center space-x-2 text-slate-400 text-xs py-1">
                <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                <span>Architect is reasoning...</span>
              </div>
            )}
          </div>

          {/* Chat Dock Footer / Input Area */}
          <div className="p-3 border-t border-slate-200 bg-white">
            <form onSubmit={(e) => handleSendChat(e)} className="space-y-2">
              <div className="flex items-center justify-between pb-1">
                {/* Model Selector Pill */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setIsModelDropdownOpen((prev) => !prev)}
                    className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-md bg-white border border-slate-200 text-[10px] font-semibold text-slate-700 hover:bg-slate-100 transition-colors shadow-2xs"
                  >
                    <Sparkles className="w-2.5 h-2.5 text-indigo-600" />
                    <span>{AVAILABLE_MODELS.find((m) => m.id === selectedModel)?.name}</span>
                    <ChevronDown className="w-2.5 h-2.5 text-slate-400" />
                  </button>

                  {isModelDropdownOpen && (
                    <div className="absolute bottom-6 left-0 w-48 bg-white border border-slate-200 rounded-xl shadow-lg z-50 p-1 space-y-0.5 animate-in fade-in slide-in-from-bottom-2">
                      {AVAILABLE_MODELS.map((m) => {
                        const IconComp = m.icon;
                        const isCur = selectedModel === m.id;
                        return (
                          <button
                            key={m.id}
                            type="button"
                            onClick={() => {
                              setSelectedModel(m.id);
                              setIsModelDropdownOpen(false);
                            }}
                            className={`w-full text-left px-2 py-1.5 rounded-lg text-[11px] flex items-center justify-between transition-colors ${
                              isCur
                                ? "bg-indigo-50 text-indigo-700 font-semibold"
                                : "text-slate-700 hover:bg-slate-100"
                            }`}
                          >
                            <div className="flex items-center space-x-1.5">
                              <IconComp className="w-3 h-3 text-indigo-600" />
                              <span>{m.name}</span>
                            </div>
                            {isCur && <Check className="w-3 h-3 text-indigo-600" />}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

                <span className="text-[10px] text-slate-400 font-mono">ChromaDB RAG Active</span>
              </div>

              {studioStage === "code_studio" && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar">
                  <button
                    type="button"
                    onClick={() => handleSendCodeFileToChat(activeCodeFile)}
                    className="shrink-0 text-[10px] px-2 py-0.5 rounded-md bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-medium transition-colors border border-indigo-200/50"
                  >
                    Explain {activeCodeFile}
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleSendChat(
                        undefined,
                        "How are the multiple datasets harmonized to prevent data leakage and format mismatches?"
                      )
                    }
                    className="shrink-0 text-[10px] px-2 py-0.5 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
                  >
                    Harmonization logic?
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      handleSendChat(
                        undefined,
                        "How do I run the full pipeline in PowerShell and what GPU resources are needed?"
                      )
                    }
                    className="shrink-0 text-[10px] px-2 py-0.5 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
                  >
                    Execution guide?
                  </button>
                </div>
              )}

              {/* Text Input & Submit Button */}
              <div className="flex items-end space-x-2">
                <textarea
                  rows={2}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendChat(e);
                    }
                  }}
                  placeholder="Ask a question or request research plan edits..."
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:bg-white focus:ring-1 focus:ring-indigo-500 leading-relaxed resize-none"
                />

                <button
                  type="submit"
                  disabled={!chatInput.trim() || isSendingChat}
                  className="w-8 h-8 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white flex items-center justify-center shrink-0 transition-colors shadow-xs"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>
        </aside>

        {/* ======================================================== */}
        {/* RIGHT PANEL (80%): Dynamic Orchestrator Workspace        */}
        {/* ======================================================== */}
        <main className="flex-1 flex flex-col overflow-y-auto bg-slate-50/60 p-6 lg:p-10">
          {/* Notice banner if present */}
          {submitNotice && (
            <div className="mb-6 p-4 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-900 text-sm flex items-start space-x-3 transition-all animate-in fade-in slide-in-from-top-2 shadow-xs">
              <CheckCircle2 className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">{submitNotice}</p>
                <p className="text-xs text-indigo-700 mt-0.5">
                  Your choices are indexed persistently in the local vector store.
                </p>
              </div>
            </div>
          )}

          {/* ========================================== */}
          {/* VIEW 1: REQUIREMENTS QUESTIONNAIRE         */}
          {/* ========================================== */}
          {studioStage === "questionnaire" && (
            <>
              {loadingQuestions ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4">
                  <div className="w-10 h-10 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                  <div className="text-center space-y-1">
                    <h3 className="font-bold text-slate-800 text-base">
                      Synthesizing Requirements Questionnaire...
                    </h3>
                    <p className="text-xs text-slate-500 max-w-sm">
                      The local AI architect is formulating tailored questions covering compute, datasets, algorithms, and serving latency.
                    </p>
                  </div>
                </div>
              ) : !isFeasible ? (
                <div className="max-w-2xl w-full mx-auto my-auto bg-white rounded-3xl border border-amber-200/90 p-8 sm:p-10 shadow-sm space-y-6">
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 rounded-2xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center shrink-0">
                      <AlertTriangle className="w-6 h-6" />
                    </div>
                    <div className="space-y-1">
                      <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800">
                        Pretraining Feasibility Advisory
                      </span>
                      <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
                        Sorry, this project cannot be trained from scratch as requested.
                      </h2>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200/70 text-slate-800 text-sm leading-relaxed space-y-2">
                    <p className="font-medium text-slate-900">{sorryMessage}</p>
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3">
                    <Link
                      href="/projects/new"
                      className="w-full sm:w-auto px-5 py-2.5 rounded-xl border border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-semibold text-center transition-colors"
                    >
                      ← Back to Project Creation
                    </Link>
                  </div>
                </div>
              ) : (
                <div className="max-w-3xl w-full mx-auto space-y-6">
                  {/* Progress Header */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs flex items-center justify-between">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200/50">
                          {currentQ?.category || "Architectural Dimension"}
                        </span>
                        <span className="text-xs text-slate-400">•</span>
                        <span className="text-xs text-slate-500">
                          Question {currentQuestionIndex + 1} of {questions.length}
                        </span>
                      </div>
                      <h2 className="text-lg font-bold text-slate-900 mt-1">
                        Requirements Questionnaire (Training from Scratch)
                      </h2>
                    </div>

                    <div className="w-32 hidden sm:block">
                      <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
                        <span>Progress</span>
                        <span className="font-bold text-slate-700">
                          {Math.round(((currentQuestionIndex + 1) / questions.length) * 100)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-600 transition-all duration-300 rounded-full"
                          style={{
                            width: `${((currentQuestionIndex + 1) / questions.length) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Main Question Card */}
                  <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6">
                    <div>
                      <h3 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight leading-snug">
                        {currentQ?.question}
                      </h3>
                      {currentQ?.context && (
                        <div className="mt-2.5 p-3 rounded-xl bg-slate-50 border border-slate-100 flex items-start space-x-2 text-xs text-slate-600">
                          <Info className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
                          <p>
                            <span className="font-semibold text-slate-800">Why this matters: </span>
                            {currentQ.context}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Pre-Generated Options */}
                    <div className="space-y-3">
                      <label className="text-xs font-bold uppercase tracking-wider text-slate-500">
                        Select an option:
                      </label>
                      <div className="space-y-2">
                        {currentQ?.options?.map((opt, oIdx) => {
                          const isSelected = currentAnswer.option === opt;
                          const isAiRec = currentQ.ai_recommendation === opt;
                          return (
                            <button
                              key={oIdx}
                              type="button"
                              onClick={() => handleSelectOption(opt)}
                              className={`w-full text-left p-4 rounded-2xl border text-xs sm:text-sm transition-all flex items-start justify-between group ${
                                isSelected
                                  ? "border-indigo-600 bg-indigo-50/50 shadow-xs font-semibold text-indigo-950 ring-1 ring-indigo-500"
                                  : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/80 text-slate-800"
                              }`}
                            >
                              <div className="space-y-1 pr-4">
                                <div>{opt}</div>
                                {isAiRec && (
                                  <span className="inline-flex items-center space-x-1 text-[10px] font-bold text-indigo-700 bg-indigo-100/80 px-2 py-0.5 rounded-full">
                                    <Sparkles className="w-2.5 h-2.5" />
                                    <span>AI Preferred</span>
                                  </span>
                                )}
                              </div>
                              <div
                                className={`w-5 h-5 rounded-full border flex items-center justify-center shrink-0 mt-0.5 transition-colors ${
                                  isSelected
                                    ? "border-indigo-600 bg-indigo-600 text-white"
                                    : "border-slate-300 group-hover:border-slate-400"
                                }`}
                              >
                                {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Custom Input */}
                    <div className="pt-2 border-t border-slate-100 space-y-2">
                      <label className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                        <Edit3 className="w-3.5 h-3.5" />
                        <span>Or write custom requirements:</span>
                      </label>
                      <input
                        type="text"
                        value={currentAnswer.custom || ""}
                        onChange={(e) => handleCustomTextChange(e.target.value)}
                        placeholder="Type anything specific (e.g. custom loss weights, distributed nodes, proprietary dataset format)..."
                        className="w-full px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all"
                      />
                    </div>
                  </div>

                  {/* Bottom Navigation & Actions */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-4 sm:p-5 flex items-center justify-between shadow-xs">
                    <button
                      type="button"
                      onClick={handlePrev}
                      disabled={currentQuestionIndex === 0}
                      className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:text-slate-900 hover:bg-slate-50 text-xs font-semibold disabled:opacity-30 disabled:pointer-events-none transition-colors"
                    >
                      ← Previous
                    </button>

                    <button
                      type="button"
                      onClick={handleSkipOrAiDefault}
                      className="hidden sm:inline-flex items-center space-x-1.5 text-xs text-indigo-600 hover:text-indigo-800 font-medium px-3 py-1.5 rounded-lg hover:bg-indigo-50 transition-colors"
                    >
                      <Lightbulb className="w-3.5 h-3.5 text-indigo-500" />
                      <span>Use AI Default & Skip</span>
                    </button>

                    {currentQuestionIndex < questions.length - 1 ? (
                      <button
                        type="button"
                        onClick={handleNext}
                        className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-xs transition-colors"
                      >
                        Next Question →
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={handleSubmitQuestionnaire}
                        className="inline-flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-sm transition-all"
                      >
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span>Submit Questionnaire →</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ========================================== */}
          {/* VIEW 2: DISCOVERED PRODUCTION DATASETS     */}
          {/* ========================================== */}
          {studioStage === "datasets" && (
            <div className="max-w-4xl w-full mx-auto space-y-6 animate-in fade-in">
              {/* Header Banner */}
              <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Phase 2A: Discovered Production Datasets
                    </span>
                    <span className="text-xs text-slate-500">
                      • {datasets.length} Verified Sources Found
                    </span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900">
                    Suitable Datasets for Training from Scratch
                  </h2>
                  <p className="text-xs text-slate-500">
                    The agentic orchestrator verified licenses, modalities, and sample sizes for your project.
                  </p>
                </div>

                <div className="flex items-center space-x-2 shrink-0">
                  <span className="text-xs font-semibold px-3 py-1.5 rounded-xl bg-indigo-50 text-indigo-700 border border-indigo-200">
                    {selectedDatasetIds.length} Selected
                  </span>
                </div>
              </div>

              {loadingDatasets ? (
                <div className="bg-white rounded-3xl border border-slate-200 p-12 flex flex-col items-center justify-center space-y-4">
                  <div className="w-10 h-10 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                  <p className="text-sm font-bold text-slate-800">
                    Searching scientific repositories & verifying licenses...
                  </p>
                  <p className="text-xs text-slate-500">
                    Scanning Synapse, TCIA, Kaggle, HuggingFace, and PhysioNet.
                  </p>
                </div>
              ) : (
                /* Dataset Cards Grid (5-7 Datasets) */
                <div className="space-y-4">
                  {datasets.map((ds) => {
                    const isSelected = selectedDatasetIds.includes(ds.id);
                    return (
                      <div
                        key={ds.id}
                        className={`bg-white rounded-3xl border p-5 sm:p-6 transition-all shadow-xs ${
                          isSelected
                            ? "border-indigo-600 ring-2 ring-indigo-500/20 bg-indigo-50/20"
                            : "border-slate-200 hover:border-slate-300"
                        }`}
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
                          <div className="space-y-1">
                            <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                              <h3 className="font-extrabold text-slate-900 text-sm sm:text-base">
                                {ds.title}
                              </h3>
                              {ds.recommended && (
                                <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800 text-[10px] font-bold">
                                  <Sparkles className="w-2.5 h-2.5" />
                                  <span>Recommended</span>
                                </span>
                              )}
                              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold flex items-center space-x-1">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>Verified</span>
                              </span>
                            </div>
                            <p className="text-xs text-slate-500 flex items-center space-x-1">
                              <Globe className="w-3 h-3 text-slate-400" />
                              <span>{ds.source}</span>
                            </p>
                          </div>

                          {/* Suitability Score */}
                          <div className="flex items-center space-x-2">
                            <div className="text-right">
                              <div className="text-xs font-bold text-slate-900">
                                {ds.suitability_score}% Match
                              </div>
                              <div className="text-[10px] text-slate-400">Suitability</div>
                            </div>
                            <div className="w-10 h-10 rounded-full bg-indigo-50 border border-indigo-200 flex items-center justify-center font-extrabold text-xs text-indigo-700">
                              {ds.suitability_score}
                            </div>
                          </div>
                        </div>

                        {/* Specs & Modality Row */}
                        <div className="py-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                            <span className="text-[10px] text-slate-400 block font-semibold">MODALITY</span>
                            <span className="font-medium text-slate-800 text-[11px] truncate block">
                              {ds.modality}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                            <span className="text-[10px] text-slate-400 block font-semibold">FORMAT</span>
                            <span className="font-medium text-slate-800 text-[11px] truncate block">
                              {ds.format}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                            <span className="text-[10px] text-slate-400 block font-semibold">SAMPLE COUNT</span>
                            <span className="font-medium text-slate-800 text-[11px] truncate block">
                              {ds.samples}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                            <span className="text-[10px] text-slate-400 block font-semibold">LICENSE</span>
                            <span className="font-medium text-slate-800 text-[11px] truncate block">
                              {ds.license}
                            </span>
                          </div>
                        </div>

                        {/* Suitability Reasoning */}
                        <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-xs text-slate-600 mb-4">
                          <span className="font-semibold text-slate-800">Architect Analysis: </span>
                          {ds.suitability_reason}
                        </div>

                        {/* Card Actions */}
                        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
                          <div className="flex items-center space-x-2">
                            {/* Select for Training button */}
                            <button
                              type="button"
                              onClick={() => handleToggleDataset(ds.id)}
                              className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition-colors ${
                                isSelected
                                  ? "bg-indigo-600 text-white shadow-xs"
                                  : "border border-slate-200 text-slate-700 hover:bg-slate-50"
                              }`}
                            >
                              {isSelected ? (
                                <>
                                  <Check className="w-3.5 h-3.5 stroke-[3]" />
                                  <span>Selected for Training</span>
                                </>
                              ) : (
                                <>
                                  <Plus className="w-3.5 h-3.5" />
                                  <span>Select for Training</span>
                                </>
                              )}
                            </button>

                            {/* Send to Assistant button */}
                            <button
                              type="button"
                              onClick={() => handleSendDatasetToChat(ds)}
                              className="px-3 py-1.5 rounded-xl border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-xs font-medium flex items-center space-x-1.5 transition-colors"
                            >
                              <Bot className="w-3.5 h-3.5 text-indigo-600" />
                              <span>Send to Assistant</span>
                            </button>
                          </div>

                          {/* View Source Website Button */}
                          <button
                            type="button"
                            onClick={() => handleViewDatasetSource(ds)}
                            className="px-3.5 py-1.5 rounded-xl border border-slate-200 text-indigo-600 hover:bg-indigo-50 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                          >
                            <span>View Source</span>
                            <ExternalLink className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Bottom Navigation */}
              <div className="bg-white rounded-2xl border border-slate-200 p-4 sm:p-5 flex items-center justify-between shadow-xs">
                <button
                  type="button"
                  onClick={() => setStudioStage("questionnaire")}
                  className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:text-slate-900 hover:bg-slate-50 text-xs font-semibold transition-colors"
                >
                  ← Back to Questionnaire
                </button>

                <button
                  type="button"
                  onClick={handleGenerateTechResearch}
                  disabled={selectedDatasetIds.length === 0 || loadingTechResearch}
                  className="inline-flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-xs font-semibold shadow-xs transition-colors"
                >
                  <span>Generate Technical Blueprint</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* ========================================== */}
          {/* VIEW 3: IN-PANEL DATASET SOURCE PREVIEW    */}
          {/* ========================================== */}
          {studioStage === "dataset_preview" && activePreviewDataset && (
            <div className="max-w-5xl w-full mx-auto space-y-4 animate-in fade-in flex-1 flex flex-col">
              {/* Back Bar */}
              <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <button
                    type="button"
                    onClick={() => setStudioStage("datasets")}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-bold transition-colors"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>Back to Datasets</span>
                  </button>
                  <span className="text-slate-300">|</span>
                  <div className="text-xs font-bold text-slate-900 truncate max-w-md">
                    {activePreviewDataset.title}
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => handleToggleDataset(activePreviewDataset.id)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1 transition-colors ${
                      selectedDatasetIds.includes(activePreviewDataset.id)
                        ? "bg-indigo-600 text-white"
                        : "border border-slate-200 text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    {selectedDatasetIds.includes(activePreviewDataset.id) ? (
                      <>
                        <Check className="w-3.5 h-3.5 stroke-[3]" />
                        <span>Selected</span>
                      </>
                    ) : (
                      <>
                        <Plus className="w-3.5 h-3.5" />
                        <span>Select for Training</span>
                      </>
                    )}
                  </button>

                  <a
                    href={activePreviewDataset.url}
                    target="_blank"
                    rel="noreferrer"
                    className="p-1.5 rounded-xl border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition-colors"
                    title="Open in new window"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>

              {/* Source Details & Live Portal Card */}
              <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6 flex-1 flex flex-col">
                <div className="space-y-2 border-b border-slate-100 pb-5">
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200">
                      Official Source Repository
                    </span>
                    <span className="text-xs text-slate-500">• {activePreviewDataset.source}</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-slate-900">
                    {activePreviewDataset.title}
                  </h2>
                  <p className="text-xs text-slate-600 leading-relaxed max-w-3xl">
                    {activePreviewDataset.suitability_reason}
                  </p>
                </div>

                {/* Metadata Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                    <div className="text-[10px] font-bold text-slate-400 uppercase">Acquisition Modality</div>
                    <div className="font-semibold text-slate-900 mt-1">{activePreviewDataset.modality}</div>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                    <div className="text-[10px] font-bold text-slate-400 uppercase">File Packaging</div>
                    <div className="font-semibold text-slate-900 mt-1">{activePreviewDataset.format}</div>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                    <div className="text-[10px] font-bold text-slate-400 uppercase">Cohort Size</div>
                    <div className="font-semibold text-slate-900 mt-1">{activePreviewDataset.samples}</div>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                    <div className="text-[10px] font-bold text-slate-400 uppercase">Permissive License</div>
                    <div className="font-semibold text-slate-900 mt-1">{activePreviewDataset.license}</div>
                  </div>
                </div>

                {/* Embedded Frame or Repository Card */}
                <div className="flex-1 rounded-2xl border border-slate-200 overflow-hidden bg-slate-50/50 flex flex-col items-center justify-center p-8 text-center space-y-4">
                  <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-200 text-indigo-600 flex items-center justify-center">
                    <Globe className="w-7 h-7" />
                  </div>
                  <div className="max-w-md space-y-1">
                    <h4 className="font-bold text-slate-900 text-sm">
                      Repository Portal: {activePreviewDataset.source}
                    </h4>
                    <p className="text-xs text-slate-500">
                      Direct connection to verified source archive. You can access documentation, manifests, and download links directly.
                    </p>
                  </div>
                  <a
                    href={activePreviewDataset.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-xs transition-colors"
                  >
                    <span>Launch Official Dataset Portal</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
            </div>
          )}

          {/* ========================================== */}
          {/* VIEW 4: TECHNICAL ARCHITECTURE BLUEPRINT   */}
          {/* ========================================== */}
          {studioStage === "tech_research" && (
            <div className="max-w-4xl w-full mx-auto space-y-6 animate-in fade-in">
              {loadingTechResearch ? (
                <div className="bg-white rounded-3xl border border-slate-200 p-12 flex flex-col items-center justify-center space-y-4">
                  <div className="w-10 h-10 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                  <p className="text-sm font-bold text-slate-800">
                    Synthesizing Neural Backbones, Loss Equations, & Compute Sizing...
                  </p>
                  <p className="text-xs text-slate-500">
                    Analyzing inductive biases and formulating implementation tasks from scratch.
                  </p>
                </div>
              ) : techResearch ? (
                <>
                  {/* Blueprint Header */}
                  <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
                          Phase 2B: Architecture & Technical Blueprint
                        </span>
                        <span className="text-xs text-slate-500">• Training From Scratch</span>
                      </div>
                      <h2 className="text-xl sm:text-2xl font-black text-slate-900">
                        {techResearch.title}
                      </h2>
                      <p className="text-xs text-slate-500">
                        Comprehensive architecture specifications, loss objectives, and execution tasks.
                      </p>
                    </div>

                    <div className="flex items-center space-x-2 shrink-0">
                      <button
                        type="button"
                        onClick={() => setIsEditingTechResearch((prev) => !prev)}
                        className={`px-3 py-1.5 rounded-xl border text-xs font-semibold flex items-center space-x-1.5 transition-colors ${
                          isEditingTechResearch
                            ? "bg-amber-50 border-amber-300 text-amber-800"
                            : "border-slate-200 text-slate-700 hover:bg-slate-50"
                        }`}
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                        <span>{isEditingTechResearch ? "Done Editing" : "Edit Plan"}</span>
                      </button>
                    </div>
                  </div>

                  {/* Section 1: Architecture & Neural Topology */}
                  <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 shadow-xs space-y-5">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <div className="flex items-center space-x-2">
                        <Layers className="w-4 h-4 text-indigo-600" />
                        <h3 className="font-extrabold text-sm sm:text-base text-slate-900">
                          1. Model Backbone Architecture
                        </h3>
                      </div>
                      <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700">
                        {techResearch.architecture.selected_name}
                      </span>
                    </div>

                    <p className="text-xs text-slate-600 leading-relaxed">
                      {techResearch.architecture.rationale}
                    </p>

                    {/* Layer Breakdown Box */}
                    <div className="space-y-1.5">
                      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                        Layer Topology Breakdown:
                      </div>
                      <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 font-mono text-[11px] text-slate-800 space-y-1">
                        {techResearch.architecture.layer_breakdown.map((l, i) => (
                          <div key={i} className="leading-relaxed">
                            {l}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Alternatives Evaluated */}
                    <div className="space-y-2 pt-2">
                      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                        Evaluated Alternatives:
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {techResearch.architecture.alternatives_evaluated.map((alt, i) => (
                          <div key={i} className="p-3.5 rounded-2xl border border-slate-200 bg-slate-50/50 space-y-1.5 text-xs">
                            <span className="font-bold text-slate-900">{alt.name}</span>
                            <div className="text-emerald-700 text-[11px]">
                              <span className="font-semibold">Pros: </span>{alt.pros}
                            </div>
                            <div className="text-rose-700 text-[11px]">
                              <span className="font-semibold">Cons: </span>{alt.cons}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Section 2: Preprocessing & Loss Formulations */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    {/* Preprocessing */}
                    <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs space-y-4">
                      <div className="flex items-center space-x-2 border-b border-slate-100 pb-3">
                        <Sliders className="w-4 h-4 text-indigo-600" />
                        <h3 className="font-extrabold text-sm text-slate-900">
                          2. Preprocessing & Normalization
                        </h3>
                      </div>
                      <div className="space-y-2 text-xs text-slate-700">
                        {techResearch.preprocessing.steps.map((step, i) => (
                          <div key={i} className="flex items-start space-x-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 mt-1.5 shrink-0" />
                            <span>{step}</span>
                          </div>
                        ))}
                      </div>
                      <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-[11px] font-mono text-slate-700">
                        <span className="font-bold">Formula: </span>{techResearch.preprocessing.normalization_formula}
                      </div>
                    </div>

                    {/* Loss Objectives */}
                    <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs space-y-4">
                      <div className="flex items-center space-x-2 border-b border-slate-100 pb-3">
                        <Zap className="w-4 h-4 text-indigo-600" />
                        <h3 className="font-extrabold text-sm text-slate-900">
                          3. Loss Formulation & Objectives
                        </h3>
                      </div>
                      <div className="text-xs font-bold text-slate-900">
                        {techResearch.loss_formulation.primary_loss}
                      </div>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        {techResearch.loss_formulation.formula_description}
                      </p>
                      <div className="p-3 rounded-xl bg-indigo-50/60 border border-indigo-100 text-[11px] text-indigo-900 space-y-1">
                        <div><span className="font-semibold">Weights: </span>{techResearch.loss_formulation.class_weights}</div>
                        <div><span className="font-semibold">Supervision: </span>{techResearch.loss_formulation.deep_supervision_weights}</div>
                      </div>
                    </div>
                  </div>

                  {/* Section 3: Hardware & Compute Sizing */}
                  <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 shadow-xs space-y-4">
                    <div className="flex items-center space-x-2 border-b border-slate-100 pb-3">
                      <Cpu className="w-4 h-4 text-indigo-600" />
                      <h3 className="font-extrabold text-sm sm:text-base text-slate-900">
                        4. Compute & Hardware Footprint Estimation
                      </h3>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">MODEL SIZE</span>
                        <span className="font-extrabold text-slate-900 text-sm">{techResearch.hardware_and_compute.estimated_parameters}</span>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">MEMORY ESTIMATE</span>
                        <span className="font-semibold text-slate-900 text-xs">{techResearch.hardware_and_compute.estimated_vram_footprint}</span>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">BATCH SIZING</span>
                        <span className="font-semibold text-slate-900 text-xs">{techResearch.hardware_and_compute.recommended_batch_size}</span>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">GRAD ACCUMULATION</span>
                        <span className="font-semibold text-slate-900 text-xs">{techResearch.hardware_and_compute.gradient_accumulation_steps}</span>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">PRECISION</span>
                        <span className="font-semibold text-slate-900 text-xs">{techResearch.hardware_and_compute.mixed_precision}</span>
                      </div>
                      <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
                        <span className="text-[10px] text-slate-400 font-bold block uppercase">DURATION ESTIMATE</span>
                        <span className="font-semibold text-slate-900 text-xs">{techResearch.hardware_and_compute.estimated_epochs_and_duration}</span>
                      </div>
                    </div>
                  </div>

                  {/* Section 4: Implementation Checklist (TODOs) */}
                  <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 shadow-xs space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <div className="flex items-center space-x-2">
                        <CheckSquare className="w-4 h-4 text-indigo-600" />
                        <h3 className="font-extrabold text-sm sm:text-base text-slate-900">
                          5. Actionable Implementation Checklist (TODOs)
                        </h3>
                      </div>
                      <button
                        type="button"
                        onClick={handleAddTodo}
                        className="px-2.5 py-1 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-semibold flex items-center space-x-1 transition-colors"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Task</span>
                      </button>
                    </div>

                    <div className="space-y-2.5">
                      {techResearch.implementation_todos.map((todo) => {
                        const isDone = todo.status === "completed";
                        return (
                          <div
                            key={todo.id}
                            className={`p-3.5 rounded-2xl border transition-all flex items-start space-x-3 ${
                              isDone
                                ? "bg-emerald-50/40 border-emerald-200/70"
                                : "bg-slate-50 border-slate-200/80"
                            }`}
                          >
                            <button
                              type="button"
                              onClick={() => handleToggleTodo(todo.id)}
                              className="mt-0.5 text-slate-400 hover:text-indigo-600 transition-colors shrink-0"
                            >
                              {isDone ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                              ) : (
                                <Square className="w-4 h-4" />
                              )}
                            </button>

                            <div className="space-y-0.5 flex-1 text-xs">
                              <span
                                className={`font-bold block ${
                                  isDone ? "text-slate-500 line-through" : "text-slate-900"
                                }`}
                              >
                                {todo.title}
                              </span>
                              <span className="text-slate-500 block">{todo.description}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Bottom Finalize & Submit Bar */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-5 flex flex-wrap items-center justify-between gap-3 shadow-xs">
                    <button
                      type="button"
                      onClick={() => setStudioStage("datasets")}
                      className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:text-slate-900 hover:bg-slate-50 text-xs font-semibold transition-colors"
                    >
                      ← Back to Datasets
                    </button>

                    <div className="flex items-center space-x-3">
                      <button
                        type="button"
                        onClick={handleSubmitResearchPlan}
                        className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-colors"
                      >
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        <span>Save Research Plan</span>
                      </button>

                      <button
                        type="button"
                        onClick={handleGenerateCodePipeline}
                        disabled={loadingCode}
                        className="inline-flex items-center space-x-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-bold shadow-sm transition-all"
                      >
                        <FileCode className="w-4 h-4 text-white" />
                        <span>Generate & Save Code Pipeline →</span>
                      </button>
                    </div>
                  </div>
                </>
              ) : null}
            </div>
          )}

          {/* ========================================== */}
          {/* VIEW 4: CODE STUDIO & WORKSPACE PIPELINE   */}
          {/* ========================================== */}
          {studioStage === "code_studio" && (
            <div className="space-y-6">
              {loadingCode ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-6 py-16">
                  <div className="relative">
                    <div className="w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <FileCode className="w-6 h-6 text-indigo-600" />
                    </div>
                  </div>
                  <div className="text-center space-y-2 max-w-md">
                    <h3 className="font-extrabold text-slate-900 text-lg">
                      Synthesizing Production Code Pipeline...
                    </h3>
                    <p className="text-xs text-slate-500 leading-relaxed">
                      Harmonizing multi-dataset geometry & tensors, generating complete PyTorch training loops, Soft Dice + Focal loss, validation scoring, and saving all files directly into <span className="font-mono text-indigo-600 font-semibold">{relativeWorkspacePath || "workspace_projects/"}</span>.
                    </p>
                  </div>

                  <div className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-4 space-y-2.5 shadow-xs text-xs">
                    <div className="flex items-center space-x-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      <span>Harmonizing multi-dataset input tensors & geometry</span>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      <span>Writing random weight initialization & custom backbone</span>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      <span>Configuring Soft Dice + Focal composite loss & AMP mixed precision</span>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      <span>Writing evaluation metrics (DSC, HD95, sensitivity)</span>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-700">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      <span>Generating Windows PowerShell runner & dependencies</span>
                    </div>
                    <div className="flex items-center space-x-2 text-indigo-600 font-medium">
                      <div className="w-3.5 h-3.5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin shrink-0" />
                      <span>Saving files & indexing into ChromaDB vector store...</span>
                    </div>
                  </div>
                </div>
              ) : Object.keys(codeFiles).length > 0 ? (
                <div className="space-y-5">
                  {/* Top Studio Toolbar */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-4 sm:p-5 flex flex-wrap items-center justify-between gap-3 shadow-xs">
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() => setStudioStage("tech_research")}
                        className="px-3 py-1.5 rounded-xl border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-xs font-semibold flex items-center space-x-1 transition-colors"
                      >
                        <ArrowLeft className="w-3.5 h-3.5" />
                        <span>Back to Blueprint</span>
                      </button>

                      <div className="h-4 w-px bg-slate-200 hidden sm:block" />

                      {/* Dedicated Workspace Folder Badge */}
                      <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-200/90 text-xs">
                        <FolderOpen className="w-4 h-4 text-indigo-600 shrink-0" />
                        <span className="text-slate-500 font-medium">Saved Location:</span>
                        <span className="font-mono font-bold text-slate-800 text-[11px] truncate max-w-[260px] sm:max-w-none">
                          {relativeWorkspacePath || `workspace_projects/${projectId}`}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleCopyCode(workspacePath || relativeWorkspacePath)}
                          title="Copy absolute folder path"
                          className="text-slate-400 hover:text-slate-700 transition-colors p-0.5"
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200/60 flex items-center space-x-1">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>{Object.keys(codeFiles).length} Files Ready</span>
                      </span>

                      {isIndexedInRag && (
                        <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200/60 flex items-center space-x-1">
                          <Sparkles className="w-3 h-3 text-indigo-600" />
                          <span>ChromaDB RAG Synced</span>
                        </span>
                      )}
                    </div>

                    {/* Right Action Buttons */}
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={() => setShowGitHubModal(true)}
                        className="px-3.5 py-1.5 rounded-xl border border-slate-300/80 bg-white hover:bg-slate-50 text-slate-800 text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-2xs"
                      >
                        <GitBranch className="w-3.5 h-3.5 text-indigo-600" />
                        <span>Push to GitHub</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => setShowRunGuideModal(true)}
                        className="px-3.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                      >
                        <Terminal className="w-3.5 h-3.5 text-indigo-600" />
                        <span>Run Guide</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDownloadFile(activeCodeFile, codeFiles[activeCodeFile] || "")}
                        className="px-3.5 py-1.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleCopyCode(codeFiles[activeCodeFile] || "")}
                        className="px-4 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold flex items-center space-x-1.5 transition-all shadow-xs"
                      >
                        {copiedFile ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-white" />
                            <span>Copied!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy File</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Real-Time Execution Status Tracker Card */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-4 sm:p-5 shadow-xs space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                      <div className="flex items-center space-x-2.5">
                        <div
                          className={`w-2.5 h-2.5 rounded-full ${
                            executionStatus?.status === "COMPLETED"
                              ? "bg-emerald-500"
                              : executionStatus?.status === "EXECUTING"
                              ? "bg-amber-500 animate-pulse"
                              : "bg-emerald-500 animate-pulse"
                          }`}
                        />
                        <div className="flex items-center space-x-2">
                          <span className="font-extrabold text-xs sm:text-sm text-slate-900">
                            Pipeline Execution Status:
                          </span>
                          <span
                            className={`text-xs px-2.5 py-0.5 rounded-full font-bold border ${
                              executionStatus?.status === "COMPLETED"
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                : executionStatus?.status === "EXECUTING"
                                ? "bg-amber-50 text-amber-700 border-amber-200"
                                : "bg-emerald-50 text-emerald-700 border-emerald-200"
                            }`}
                          >
                            {executionStatus?.status === "COMPLETED"
                              ? "COMPLETED (100%)"
                              : executionStatus?.status === "EXECUTING"
                              ? `EXECUTING (${executionStatus.overall_progress_pct || 0}%)`
                              : "NEW — READY TO EXECUTE"}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          type="button"
                          onClick={fetchExecutionStatus}
                          disabled={isRefreshingStatus}
                          className="px-2.5 py-1 rounded-lg border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-[11px] font-semibold flex items-center space-x-1 transition-colors"
                        >
                          <RefreshCw className={`w-3 h-3 ${isRefreshingStatus ? "animate-spin" : ""}`} />
                          <span>Check Live Status</span>
                        </button>
                        <span className="text-[10px] text-slate-400 font-mono">
                          EXECUTION_STATUS.json
                        </span>
                      </div>
                    </div>

                    {/* Overall Progress Bar */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-slate-600">
                        <span className="font-medium">Overall Pipeline Convergence:</span>
                        <span className="font-bold font-mono text-slate-900">
                          {executionStatus?.overall_progress_pct || 0}%
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden border border-slate-200/60">
                        <div
                          className="h-full bg-linear-to-r from-indigo-500 to-emerald-500 rounded-full transition-all duration-500"
                          style={{ width: `${executionStatus?.overall_progress_pct || 0}%` }}
                        />
                      </div>
                    </div>

                    {/* Step-by-Step Stage Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 pt-1">
                      {(executionStatus?.steps || [
                        { id: 1, name: "download_datasets", label: "1. Download Data", status: "PENDING", progress_pct: 0 },
                        { id: 2, name: "eda_and_harmonization", label: "2. Harmonization", status: "PENDING", progress_pct: 0 },
                        { id: 3, name: "train", label: "3. Pretrain", status: "PENDING", progress_pct: 0 },
                        { id: 4, name: "evaluate_and_score", label: "4. Evaluation", status: "PENDING", progress_pct: 0 },
                        { id: 5, name: "visualize", label: "5. Visualizer", status: "PENDING", progress_pct: 0 },
                      ]).map((step: any) => {
                        const isDone = step.status === "COMPLETED";
                        const isRunning = step.status === "IN_PROGRESS" || step.status === "EXECUTING";
                        return (
                          <div
                            key={step.id}
                            className={`p-2.5 rounded-xl border text-[11px] transition-all flex flex-col justify-between ${
                              isDone
                                ? "bg-emerald-50/60 border-emerald-200 text-emerald-900"
                                : isRunning
                                ? "bg-amber-50/70 border-amber-300 text-amber-900"
                                : "bg-slate-50 border-slate-200/80 text-slate-600"
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-bold truncate">{step.label || step.name}</span>
                              {isDone ? (
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                              ) : isRunning ? (
                                <div className="w-3 h-3 border-2 border-amber-600 border-t-transparent rounded-full animate-spin shrink-0" />
                              ) : (
                                <Clock className="w-3.5 h-3.5 text-slate-300 shrink-0" />
                              )}
                            </div>
                            <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                              <span className="capitalize">{step.status.toLowerCase()}</span>
                              <span className="font-bold">{step.progress_pct || 0}%</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Multi-Dataset Harmonization Banner */}
                  <div className="bg-linear-to-r from-indigo-50/90 via-slate-50 to-blue-50/70 rounded-2xl border border-indigo-100 p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-2xs">
                    <div className="flex items-center space-x-3">
                      <div className="w-9 h-9 rounded-xl bg-white border border-indigo-200 text-indigo-600 flex items-center justify-center shrink-0 shadow-2xs">
                        <Database className="w-4 h-4" />
                      </div>
                      <div>
                        <h4 className="font-bold text-xs text-slate-900 flex items-center space-x-2">
                          <span>Multi-Dataset Harmonization Active</span>
                          <span className="text-[10px] px-2 py-0.5 rounded-md bg-white border border-indigo-200 text-indigo-700 font-semibold">
                            Zero Leakage
                          </span>
                        </h4>
                        <p className="text-[11px] text-slate-600 mt-0.5">
                          Ingests {selectedDatasetIds.length > 0 ? selectedDatasetIds.length : "multi"} heterogeneous sources, resamples spatial voxel grids, and normalizes into unified tensors before batching.
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-1.5">
                      {(datasets.filter((d) => selectedDatasetIds.includes(d.id)).length > 0
                        ? datasets.filter((d) => selectedDatasetIds.includes(d.id))
                        : datasets.slice(0, 3)
                      ).map((d) => (
                        <span
                          key={d.id}
                          className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-white border border-slate-200 text-slate-700"
                        >
                          {d.title}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Main IDE Workspace: File Explorer + Code Editor */}
                  <div className="bg-white rounded-3xl border border-slate-200 overflow-hidden shadow-xs flex flex-col lg:flex-row min-h-[620px]">
                    {/* Left File Explorer List */}
                    <div className="w-full lg:w-72 border-b lg:border-b-0 lg:border-r border-slate-200 bg-slate-50/60 flex flex-col shrink-0">
                      <div className="p-3.5 border-b border-slate-200 bg-slate-100/60 flex items-center justify-between">
                        <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-600 flex items-center space-x-1.5">
                          <Folder className="w-3.5 h-3.5 text-indigo-600" />
                          <span>Workspace Files</span>
                        </span>
                        <span className="text-[10px] px-2 py-0.2 rounded-full bg-white border border-slate-200 text-slate-600 font-semibold">
                          {Object.keys(codeFiles).length} files
                        </span>
                      </div>

                      <div className="p-2 space-y-1 overflow-y-auto flex-1">
                        {Object.keys(codeFiles).map((filename) => {
                          const isActive = activeCodeFile === filename;
                          const isPython = filename.endsWith(".py");
                          const isYaml = filename.endsWith(".yaml") || filename.endsWith(".yml");
                          const isShell = filename.endsWith(".ps1") || filename.endsWith(".sh");
                          const isJson = filename.endsWith(".json");
                          const isTrain = filename === "train.py";

                          const lineCount = (codeFiles[filename] || "").split("\n").length;

                          return (
                            <button
                              key={filename}
                              type="button"
                              onClick={() => setActiveCodeFile(filename)}
                              className={`w-full text-left px-3 py-2.5 rounded-xl text-xs flex items-center justify-between transition-all ${
                                isActive
                                  ? "bg-indigo-600 text-white font-bold shadow-xs"
                                  : "text-slate-700 hover:bg-slate-200/70 font-medium"
                              }`}
                            >
                              <div className="flex items-center space-x-2.5 truncate">
                                {isPython ? (
                                  <FileCode className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-amber-600"}`} />
                                ) : isYaml ? (
                                  <Sliders className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-sky-600"}`} />
                                ) : isShell ? (
                                  <Terminal className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-emerald-600"}`} />
                                ) : isJson ? (
                                  <Database className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-indigo-600"}`} />
                                ) : (
                                  <FileText className={`w-4 h-4 shrink-0 ${isActive ? "text-white" : "text-slate-500"}`} />
                                )}
                                <span className="truncate">{filename}</span>
                              </div>

                              <div className="flex items-center space-x-1 shrink-0">
                                {isTrain ? (
                                  <span
                                    className={`text-[9px] px-1.5 py-0.2 rounded-md font-bold uppercase tracking-wider ${
                                      isActive ? "bg-white/20 text-white" : "bg-indigo-50 text-indigo-700 border border-indigo-200"
                                    }`}
                                  >
                                    Core
                                  </span>
                                ) : filename === "EXECUTION_STATUS.json" ? (
                                  <span
                                    className={`text-[9px] px-1.5 py-0.2 rounded-md font-bold uppercase tracking-wider ${
                                      isActive ? "bg-white/20 text-white" : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                    }`}
                                  >
                                    Status
                                  </span>
                                ) : filename === "download_datasets.py" ? (
                                  <span
                                    className={`text-[9px] px-1.5 py-0.2 rounded-md font-bold uppercase tracking-wider ${
                                      isActive ? "bg-white/20 text-white" : "bg-sky-50 text-sky-700 border border-sky-200"
                                    }`}
                                  >
                                    Data
                                  </span>
                                ) : filename.startsWith("push_to_github") ? (
                                  <span
                                    className={`text-[9px] px-1.5 py-0.2 rounded-md font-bold uppercase tracking-wider ${
                                      isActive ? "bg-white/20 text-white" : "bg-purple-50 text-purple-700 border border-purple-200"
                                    }`}
                                  >
                                    Git
                                  </span>
                                ) : null}
                                <span
                                  className={`text-[10px] ${
                                    isActive ? "text-indigo-100" : "text-slate-400 font-mono"
                                  }`}
                                >
                                  {lineCount}L
                                </span>
                              </div>
                            </button>
                          );
                        })}
                      </div>

                      {/* One-Click Runner Button in Explorer Footer */}
                      <div className="p-3 border-t border-slate-200 bg-white">
                        <button
                          type="button"
                          onClick={() => setShowRunGuideModal(true)}
                          className="w-full py-2 px-3 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold flex items-center justify-center space-x-2 transition-colors shadow-xs"
                        >
                          <Play className="w-3.5 h-3.5 text-emerald-400 fill-emerald-400" />
                          <span>Run Pipeline Guide</span>
                        </button>
                      </div>
                    </div>

                    {/* Right Code Viewer */}
                    <div className="flex-1 flex flex-col min-w-0 bg-slate-50/40">
                      {/* Editor Header Bar */}
                      <div className="h-11 px-4 border-b border-slate-200 bg-white flex items-center justify-between shrink-0">
                        <div className="flex items-center space-x-2.5 truncate">
                          <span className="font-mono text-xs font-bold text-slate-900">
                            {activeCodeFile}
                          </span>
                          <span className="text-[10px] px-2 py-0.5 rounded-md bg-slate-100 text-slate-600 font-mono font-medium">
                            {(codeFiles[activeCodeFile] || "").split("\n").length} lines
                          </span>
                          <span className="text-[10px] text-slate-400 hidden sm:inline">
                            • UTF-8
                          </span>
                        </div>

                        <div className="flex items-center space-x-2">
                          <button
                            type="button"
                            onClick={() => handleSendCodeFileToChat(activeCodeFile)}
                            className="px-2.5 py-1 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-[11px] font-semibold flex items-center space-x-1 transition-colors"
                          >
                            <Sparkles className="w-3 h-3 text-indigo-600" />
                            <span>Ask AI about {activeCodeFile}</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleCopyCode(codeFiles[activeCodeFile] || "")}
                            className="p-1 text-slate-400 hover:text-slate-700 transition-colors"
                            title="Copy code"
                          >
                            {copiedFile ? (
                              <Check className="w-4 h-4 text-emerald-600" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Code Editor Body with Line Numbers */}
                      <div className="flex-1 flex overflow-x-auto overflow-y-auto max-h-[640px] font-mono text-[12px] leading-relaxed bg-white">
                        {/* Line Numbers Gutter */}
                        <div className="py-4 px-3 bg-slate-100/70 border-r border-slate-200 select-none text-right text-slate-400 font-mono text-[11px] shrink-0 min-w-[48px]">
                          {(codeFiles[activeCodeFile] || "")
                            .split("\n")
                            .map((_, idx) => (
                              <div key={idx} className="leading-relaxed">
                                {idx + 1}
                              </div>
                            ))}
                        </div>

                        {/* Syntax Code Content */}
                        <pre className="py-4 px-4 text-slate-800 font-mono text-[12px] leading-relaxed whitespace-pre overflow-x-auto flex-1 select-text">
                          <code>{codeFiles[activeCodeFile] || "# File is empty."}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4 py-20 text-center">
                  <FileCode className="w-12 h-12 text-slate-300" />
                  <div className="space-y-1">
                    <h3 className="font-bold text-slate-800 text-sm">No Code Generated Yet</h3>
                    <p className="text-xs text-slate-500">
                      Click below to generate the complete runnable multi-dataset training suite.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleGenerateCodePipeline}
                    className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold transition-all shadow-xs"
                  >
                    Generate Code Pipeline Now
                  </button>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Run Pipeline Modal Guide */}
      {showRunGuideModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 max-w-xl w-full shadow-2xl space-y-5 animate-in fade-in zoom-in-95">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-2.5">
                <div className="w-9 h-9 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-xs">
                  <Terminal className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-extrabold text-slate-900 text-base">
                    Local Execution Guide
                  </h3>
                  <p className="text-xs text-slate-500">
                    Run the multi-dataset pipeline directly in Windows PowerShell
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setShowRunGuideModal(false)}
                className="w-7 h-7 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 flex items-center justify-center text-sm font-bold transition-colors"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3.5 text-xs">
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/90 space-y-1.5">
                <span className="font-bold text-slate-800 block">Step 1: Open PowerShell and Navigate to Workspace</span>
                <div className="flex items-center justify-between bg-slate-900 text-emerald-400 font-mono text-[11px] p-2.5 rounded-xl">
                  <span className="truncate">cd &quot;{workspacePath || `workspace_projects\\${projectId}`}&quot;</span>
                  <button
                    type="button"
                    onClick={() => handleCopyCode(`cd "${workspacePath || `workspace_projects\\${projectId}`}"`)}
                    className="text-slate-400 hover:text-white transition-colors ml-2 shrink-0"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/90 space-y-1.5">
                <span className="font-bold text-slate-800 block">Step 2: Install Environment Dependencies</span>
                <div className="flex items-center justify-between bg-slate-900 text-emerald-400 font-mono text-[11px] p-2.5 rounded-xl">
                  <span>pip install -r requirements.txt</span>
                  <button
                    type="button"
                    onClick={() => handleCopyCode("pip install -r requirements.txt")}
                    className="text-slate-400 hover:text-white transition-colors ml-2 shrink-0"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/90 space-y-1.5">
                <span className="font-bold text-slate-800 block">Step 3: One-Click End-to-End Pipeline</span>
                <div className="flex items-center justify-between bg-slate-900 text-emerald-400 font-mono text-[11px] p-2.5 rounded-xl">
                  <span>powershell.exe -ExecutionPolicy Bypass -File .\run_pipeline.ps1</span>
                  <button
                    type="button"
                    onClick={() => handleCopyCode("powershell.exe -ExecutionPolicy Bypass -File .\\run_pipeline.ps1")}
                    className="text-slate-400 hover:text-white transition-colors ml-2 shrink-0"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-800 text-[11px] space-y-1">
                <span className="font-bold block">Individual Pipeline Execution Steps:</span>
                <div className="space-y-0.5 font-mono text-[10px]">
                  <p>1. Ingest & Download: <code>python download_datasets.py</code></p>
                  <p>2. Harmonization: <code>python eda_and_harmonization.py</code></p>
                  <p>3. Pretraining Loop: <code>python train.py</code></p>
                  <p>4. Validation Metrics: <code>python evaluate_and_score.py</code></p>
                  <p>5. Visualizer Plots: <code>python visualize.py</code></p>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={() => setShowRunGuideModal(false)}
                className="px-5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold transition-colors"
              >
                Close Guide
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Push to GitHub Modal */}
      {showGitHubModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 max-w-xl w-full shadow-2xl space-y-5 animate-in fade-in zoom-in-95">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-2.5">
                <div className="w-9 h-9 rounded-xl bg-slate-900 text-white flex items-center justify-center shadow-xs">
                  <GitBranch className="w-5 h-5 text-indigo-400" />
                </div>
                <div>
                  <h3 className="font-extrabold text-slate-900 text-base">
                    Push Workspace to GitHub
                  </h3>
                  <p className="text-xs text-slate-500">
                    Publish this complete pretraining workspace to your remote repository
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => {
                  setShowGitHubModal(false);
                  setGitPushResult(null);
                }}
                className="w-7 h-7 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 flex items-center justify-center text-sm font-bold transition-colors"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handlePushToGitHub} className="space-y-3.5 text-xs">
              <div className="space-y-1">
                <label className="font-bold text-slate-700 block">
                  GitHub Repository URL <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="https://github.com/username/my-ml-project.git"
                  value={githubRepoUrl}
                  onChange={(e) => setGithubRepoUrl(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 placeholder-slate-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono text-[11px]"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="font-bold text-slate-700 block">
                    Branch Name
                  </label>
                  <input
                    type="text"
                    placeholder="main"
                    value={githubBranch}
                    onChange={(e) => setGithubBranch(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 font-mono text-[11px]"
                  />
                </div>

                <div className="space-y-1">
                  <label className="font-bold text-slate-700 block">
                    Personal Access Token (Optional)
                  </label>
                  <input
                    type="password"
                    placeholder="ghp_xxxxxxxxxxxx"
                    value={githubToken}
                    onChange={(e) => setGithubToken(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 font-mono text-[11px]"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="font-bold text-slate-700 block">
                  Commit Message
                </label>
                <input
                  type="text"
                  placeholder="Initial commit: Production pretraining pipeline from scratch"
                  value={githubCommitMsg}
                  onChange={(e) => setGithubCommitMsg(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 text-xs"
                />
              </div>

              {/* Status / Output Feedback */}
              {gitPushResult && (
                <div
                  className={`p-3 rounded-xl border text-xs flex items-center space-x-2 ${
                    gitPushResult.success
                      ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                      : "bg-red-50 text-red-800 border-red-200"
                  }`}
                >
                  {gitPushResult.success ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                  )}
                  <span>{gitPushResult.message}</span>
                </div>
              )}

              {/* Terminal Logs */}
              {gitPushLogs.length > 0 && (
                <div className="space-y-1">
                  <span className="text-[10px] text-slate-400 font-bold block uppercase">
                    Git Terminal Stream
                  </span>
                  <div className="p-3 rounded-xl bg-slate-950 text-emerald-400 font-mono text-[10px] max-h-32 overflow-y-auto space-y-1 leading-relaxed">
                    {gitPushLogs.map((log, idx) => (
                      <div key={idx} className="whitespace-pre-wrap">
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Alternative PowerShell Command */}
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/90 text-slate-600 text-[11px] space-y-1">
                <span className="font-bold block text-slate-800">Or push directly from terminal:</span>
                <div className="flex items-center justify-between font-mono bg-slate-900 text-emerald-400 p-2 rounded-lg text-[10px]">
                  <span className="truncate">powershell.exe -ExecutionPolicy Bypass -File .\\push_to_github.ps1 -RepoUrl &quot;{githubRepoUrl || "https://github.com/..."}&quot;</span>
                  <button
                    type="button"
                    onClick={() => handleCopyCode(`powershell.exe -ExecutionPolicy Bypass -File .\\push_to_github.ps1 -RepoUrl "${githubRepoUrl || "https://github.com/..."}"`)}
                    className="text-slate-400 hover:text-white transition-colors ml-2 shrink-0"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => {
                    setShowGitHubModal(false);
                    setGitPushResult(null);
                  }}
                  className="px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-600 text-xs font-semibold transition-colors"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={isPushingGit || !githubRepoUrl.trim()}
                  className="px-5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-bold flex items-center space-x-1.5 transition-colors shadow-xs"
                >
                  {isPushingGit ? (
                    <>
                      <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Pushing to GitHub...</span>
                    </>
                  ) : (
                    <>
                      <GitBranch className="w-3.5 h-3.5" />
                      <span>Push Workspace Now</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default function TrainerStudioPage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen flex items-center justify-center bg-white">
          <div className="w-8 h-8 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <TrainerStudioContent />
    </Suspense>
  );
}
