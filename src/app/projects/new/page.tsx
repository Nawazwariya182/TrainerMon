"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Sparkles,
  FileText,
  Image as ImageIcon,
  Mic,
  TrendingUp,
  Layers,
  Cpu,
  Target,
  Zap,
  BookOpen,
  Rocket,
  Trophy,
  Check,
  CheckCircle2,
  Activity,
  AlertCircle,
  HelpCircle,
  Bot,
  BrainCircuit,
  Eye,
} from "lucide-react";

interface BackendStatus {
  online: boolean;
  engine?: {
    loaded?: boolean;
    current_model?: string;
  };
}

export default function DescribeProjectPage() {
  const router = useRouter();

  // Form State
  const [projectName, setProjectName] = useState<string>("");
  const [projectIdea, setProjectIdea] = useState<string>("");
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [selectedApproaches, setSelectedApproaches] = useState<string[]>([]);
  const [primaryGoal, setPrimaryGoal] = useState<string>("Production application");
  const [selectedModel, setSelectedModel] = useState<string>("qwen3-8b");

  // Interaction & UI State
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [isAiRecommended, setIsAiRecommended] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Fetch backend telemetry for header status
  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetch("/api/backend-status");
        const data = await res.json();
        setBackendStatus(data);
      } catch {
        setBackendStatus({ online: false });
      }
    }
    fetchStatus();
  }, []);

  // Domain Options
  const domainOptions = [
    {
      id: "Text",
      label: "Text",
      description: "NLP, Large Language Models, reasoning, summarization, and translation",
      icon: FileText,
    },
    {
      id: "Image / CV",
      label: "Image / CV",
      description: "Computer Vision, classification, object detection, segmentation, and vision-language",
      icon: ImageIcon,
    },
    {
      id: "Audio / Speech",
      label: "Audio / Speech",
      description: "Automatic speech recognition, acoustic modeling, voice analysis, and audio synthesis",
      icon: Mic,
    },
    {
      id: "Time Series",
      label: "Time Series",
      description: "Sequential forecasting, anomaly detection, sensor telemetries, and financial modeling",
      icon: TrendingUp,
    },
  ];

  // Learning Approach Options
  const approachOptions = [
    {
      id: "Supervised Learning",
      label: "Supervised Learning",
      description: "Optimized on labeled ground-truth datasets for classification or regression tasks.",
      icon: Target,
    },
    {
      id: "Unsupervised Learning",
      label: "Unsupervised Learning",
      description: "Representation learning, clustering, dimensionality reduction, and self-supervised modeling.",
      icon: Layers,
    },
    {
      id: "Reinforcement Learning",
      label: "Reinforcement Learning",
      description: "Policy optimization, reward modeling, actor-critic agents, and feedback environments.",
      icon: Zap,
    },
    {
      id: "Deep Learning",
      label: "Deep Learning",
      description: "Multi-layer neural networks, Transformer backbones, and foundation model fine-tuning.",
      icon: Cpu,
    },
    {
      id: "Let AI choose the approach",
      label: "Let AI choose the approach",
      description: "Your assistant analyzes your idea & domains to recommend the optimal learning paradigm.",
      icon: Sparkles,
      highlight: true,
    },
  ];

  // Primary Goal Options
  const goalOptions = [
    {
      id: "Research / learning",
      title: "Research / learning",
      description: "Rapid experimentation, hypothesis testing, theoretical exploration, and paper reproduction.",
      icon: BookOpen,
      badge: "Exploration",
    },
    {
      id: "Production application",
      title: "Production application",
      description: "Optimized resource efficiency, low latency serving, high reliability SLA, and robust production deployment.",
      icon: Rocket,
      badge: "Recommended",
    },
    {
      id: "Best possible model performance",
      title: "Best possible model performance",
      description: "Maximal benchmark accuracy, largest parameter capacity, exhaustive hyperparameter sweeps.",
      icon: Trophy,
      badge: "High Precision",
    },
  ];

  // Model Selection Options
  const modelOptions = [
    {
      id: "qwen3-8b",
      name: "Qwen3-8B-Q8_0",
      description: "High-precision 8B parameter model optimized for deterministic architecture planning, stability, and fast inference.",
      icon: BrainCircuit,
      badge: "Recommended",
      ctx: "16K Context",
    },
    {
      id: "qwen3.5-9b",
      name: "Qwen3.5-9B-Q8_0",
      description: "Next-generation 9B parameter model with extended context, deep reasoning, and advanced architecture planning.",
      icon: Sparkles,
      badge: "Extended Context",
      ctx: "16K Context",
    },
    {
      id: "qwen3-vl-8b",
      name: "Qwen3-VL-8B-Instruct-Q8_0",
      description: "Vision-Language multimodal model equipped with F16 multimodal projector for visual reasoning and image inspection.",
      icon: Eye,
      badge: "Multimodal Vision",
      ctx: "Vision Projector",
    },
  ];

  // Idea Quick Suggestion Chips (Pretraining from Scratch)
  const exampleIdeas = [
    "Pretrain a 1.5B reasoning transformer from scratch on 50B synthetic tokens",
    "Train a 300M edge vision-language model from scratch on filtered image-caption pairs",
    "Pretrain a multivariate time-series foundation model from scratch on 20M sensor streams",
  ];

  // Toggle multi-select domain
  const toggleDomain = (id: string) => {
    setSelectedDomains((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  // Toggle multi-select approach
  const toggleApproach = (id: string) => {
    if (id === "Let AI choose the approach") {
      setIsAiRecommended((prev) => !prev);
      setSelectedApproaches((prev) =>
        prev.includes(id) ? prev.filter((a) => a !== id) : [...prev, id]
      );
      return;
    }

    setSelectedApproaches((prev) =>
      prev.includes(id) ? prev.filter((a) => a !== id) : [...prev, id]
    );
  };

  // Create project and route into Trainer Studio
  const handleCreateProject = async () => {
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: projectName.trim() || (projectIdea ? projectIdea.slice(0, 32) + "..." : "New ML Project"),
          idea: projectIdea.trim(),
          domains: selectedDomains,
          learningApproaches: selectedApproaches,
          primaryGoal,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      if (data.success && data.project?.id) {
        router.push(`/trainer?projectId=${data.project.id}`);
      } else {
        router.push(`/trainer?model=${selectedModel}`);
      }
    } catch (err) {
      console.error("Failed to create project:", err);
      router.push(`/trainer?model=${selectedModel}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50/70 text-slate-900 pb-20">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link
              href="/"
              className="inline-flex items-center space-x-2 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors px-2.5 py-1.5 rounded-lg hover:bg-slate-100"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Projects</span>
            </Link>
            <span className="text-slate-300">/</span>
            <span className="text-sm font-semibold text-slate-800">
              Describe your ML project
            </span>
          </div>

          {/* Telemetry status pill */}
          <div className="hidden sm:flex items-center space-x-2 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus?.online ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span className="font-medium text-slate-700">
              Inference Engine
            </span>
            <span className="text-slate-400">•</span>
            <span className="text-slate-500">
              {backendStatus?.online
                ? backendStatus.engine?.loaded
                  ? `Active • ${backendStatus.engine.current_model || "Ready"}`
                  : "Standby (Auto-offloaded)"
                : "Initializing"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Form Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-8">
        {/* Hero Section */}
        <div className="space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-200/60 text-indigo-700 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI Project Architect</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900">
            What do you want to build?
          </h1>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-3xl">
            Describe your idea. Your AI assistant will evaluate feasibility and construct a research-backed from-scratch model pretraining plan.
          </p>
        </div>

        {/* Action Notice Banner if clicked */}
        {actionNotice && (
          <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-900 text-sm flex items-start space-x-3 transition-all animate-in fade-in slide-in-from-top-2">
            <CheckCircle2 className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">{actionNotice}</p>
              <p className="text-xs text-indigo-700 mt-0.5">
                Model Domain: {selectedDomains.length ? selectedDomains.join(", ") : "None specified"} • Primary Goal: {primaryGoal}
              </p>
            </div>
          </div>
        )}

        <div className="space-y-10">
          {/* SECTION: Project Idea */}
          <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-5">
            <div className="space-y-1">
              <label
                htmlFor="project-idea"
                className="block text-base font-bold text-slate-900"
              >
                Project idea:
              </label>
              <p className="text-xs sm:text-sm text-slate-500">
                Explain your problem statement, target inputs, desired outputs, or domain requirements.
              </p>
            </div>

            {/* Optional Project Name Field */}
            <div className="space-y-1.5">
              <label htmlFor="project-name" className="text-xs font-medium text-slate-700">
                Project Name <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                id="project-name"
                type="text"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="e.g. Vision-Language Quality Inspector"
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all"
              />
            </div>

            {/* Textarea for Project Idea */}
            <div className="space-y-2">
              <textarea
                id="project-idea"
                rows={4}
                value={projectIdea}
                onChange={(e) => setProjectIdea(e.target.value)}
                placeholder="Describe your pretraining idea in plain words... (e.g., Pretrain a 1B parameter decoder-only transformer from scratch for real-time mathematical reasoning, trained on 25B deduplicated tokens with a custom BPE tokenizer.)"
                className="w-full p-4 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all leading-relaxed"
              />
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Supports natural language explanations, dataset hints, and performance criteria.</span>
                <span>{projectIdea.length} characters</span>
              </div>
            </div>

            {/* Quick Inspiration Chips */}
            <div className="pt-2 border-t border-slate-100 space-y-2">
              <div className="flex items-center space-x-1.5 text-xs text-slate-500 font-medium">
                <Bot className="w-3.5 h-3.5 text-indigo-500" />
                <span>Quick inspiration templates:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {exampleIdeas.map((example, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => {
                      setProjectIdea(example);
                      if (!projectName) {
                        setProjectName(example.split(" ").slice(0, 4).join(" "));
                      }
                    }}
                    className="text-xs text-left bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-600 px-3 py-1.5 rounded-lg border border-slate-200/70 transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </section>

          {/* SECTION 1: Model Domain */}
          <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-5">
            <div className="space-y-1">
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                1. Model Domain
              </h2>
              <p className="text-xs sm:text-sm text-slate-500">
                Select all that apply.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              {domainOptions.map((opt) => {
                const isSelected = selectedDomains.includes(opt.id);
                const IconComponent = opt.icon;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => toggleDomain(opt.id)}
                    className={`text-left p-4 rounded-xl border transition-all duration-150 flex items-start space-x-3.5 relative ${
                      isSelected
                        ? "border-indigo-600 bg-indigo-50/50 ring-2 ring-indigo-500/20 shadow-xs"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60"
                    }`}
                  >
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                        isSelected
                          ? "bg-indigo-600 text-white"
                          : "bg-slate-100 text-slate-600"
                      }`}
                    >
                      <IconComponent className="w-5 h-5" />
                    </div>

                    <div className="flex-1 pr-6 space-y-0.5">
                      <div className="font-semibold text-sm text-slate-900 flex items-center gap-1.5">
                        <span>{opt.label}</span>
                      </div>
                      <p className="text-xs text-slate-500 leading-normal">
                        {opt.description}
                      </p>
                    </div>

                    {/* Custom Checkbox Indicator */}
                    <div
                      className={`w-5 h-5 rounded-md border flex items-center justify-center absolute top-4 right-4 transition-colors ${
                        isSelected
                          ? "bg-indigo-600 border-indigo-600 text-white"
                          : "border-slate-300 bg-white"
                      }`}
                    >
                      {isSelected && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* SECTION 2: Learning Approach */}
          <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-5">
            <div className="space-y-1">
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                2. Learning Approach
              </h2>
              <p className="text-xs sm:text-sm text-slate-500">
                Select all that apply, or let AI recommend the appropriate approach.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              {approachOptions.map((opt) => {
                const isSelected = selectedApproaches.includes(opt.id);
                const IconComponent = opt.icon;
                const isSpecial = opt.highlight;

                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => toggleApproach(opt.id)}
                    className={`text-left p-4 rounded-xl border transition-all duration-150 flex items-start space-x-3.5 relative ${
                      isSpecial
                        ? isSelected
                          ? "sm:col-span-2 border-indigo-600 bg-indigo-50/70 ring-2 ring-indigo-500/30 shadow-xs"
                          : "sm:col-span-2 border-indigo-200 bg-indigo-50/20 hover:border-indigo-300 hover:bg-indigo-50/40"
                        : isSelected
                        ? "border-indigo-600 bg-indigo-50/50 ring-2 ring-indigo-500/20 shadow-xs"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60"
                    }`}
                  >
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                        isSpecial
                          ? isSelected
                            ? "bg-indigo-600 text-white"
                            : "bg-indigo-100 text-indigo-700"
                          : isSelected
                          ? "bg-indigo-600 text-white"
                          : "bg-slate-100 text-slate-600"
                      }`}
                    >
                      <IconComponent className="w-5 h-5" />
                    </div>

                    <div className="flex-1 pr-6 space-y-0.5">
                      <div className="font-semibold text-sm text-slate-900 flex items-center gap-2">
                        <span>{opt.label}</span>
                        {isSpecial && (
                          <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700">
                            AI Guided
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-500 leading-normal">
                        {opt.description}
                      </p>
                    </div>

                    {/* Custom Checkbox Indicator */}
                    <div
                      className={`w-5 h-5 rounded-md border flex items-center justify-center absolute top-4 right-4 transition-colors ${
                        isSelected
                          ? "bg-indigo-600 border-indigo-600 text-white"
                          : "border-slate-300 bg-white"
                      }`}
                    >
                      {isSelected && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* SECTION: Primary goal */}
          <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-5">
            <div className="space-y-1">
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                Primary goal
              </h2>
              <p className="text-xs sm:text-sm text-slate-500">
                Choose the focus guiding your training hyperparameters, model architecture size, and inference trade-offs.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {goalOptions.map((goal) => {
                const isSelected = primaryGoal === goal.id;
                const IconComponent = goal.icon;
                return (
                  <button
                    key={goal.id}
                    type="button"
                    onClick={() => setPrimaryGoal(goal.id)}
                    className={`text-left p-5 rounded-xl border flex flex-col justify-between transition-all duration-150 relative ${
                      isSelected
                        ? "border-indigo-600 bg-indigo-50/50 ring-2 ring-indigo-500/20 shadow-xs"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60"
                    }`}
                  >
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div
                          className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                            isSelected
                              ? "bg-indigo-600 text-white"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          <IconComponent className="w-5 h-5" />
                        </div>
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                            isSelected
                              ? "bg-indigo-100 text-indigo-700"
                              : "bg-slate-100 text-slate-500"
                          }`}
                        >
                          {goal.badge}
                        </span>
                      </div>

                      <div className="space-y-1">
                        <h3 className="font-bold text-sm text-slate-900">
                          {goal.title}
                        </h3>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          {goal.description}
                        </p>
                      </div>
                    </div>

                    {/* Radio Indicator */}
                    <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                      <span className={isSelected ? "font-semibold text-indigo-700" : "text-slate-400"}>
                        {isSelected ? "Selected goal" : "Select goal"}
                      </span>
                      <div
                        className={`w-4 h-4 rounded-full border flex items-center justify-center ${
                          isSelected
                            ? "border-indigo-600 bg-indigo-600"
                            : "border-slate-300 bg-white"
                        }`}
                      >
                        {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* SECTION: Model Selection */}
          <section className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-5">
            <div className="space-y-1">
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                Model Selection
              </h2>
              <p className="text-xs sm:text-sm text-slate-500">
                Select the base model architecture that will anchor your project and guide questionnaire generation.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {modelOptions.map((m) => {
                const isSelected = selectedModel === m.id;
                const IconComponent = m.icon;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedModel(m.id)}
                    className={`text-left p-5 rounded-xl border flex flex-col justify-between transition-all duration-150 relative ${
                      isSelected
                        ? "border-indigo-600 bg-indigo-50/50 ring-2 ring-indigo-500/20 shadow-xs"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60"
                    }`}
                  >
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div
                          className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                            isSelected
                              ? "bg-indigo-600 text-white"
                              : "bg-slate-100 text-slate-600"
                          }`}
                        >
                          <IconComponent className="w-5 h-5" />
                        </div>
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                            isSelected
                              ? "bg-indigo-100 text-indigo-700"
                              : "bg-slate-100 text-slate-500"
                          }`}
                        >
                          {m.badge}
                        </span>
                      </div>

                      <div className="space-y-1">
                        <h3 className="font-bold text-sm text-slate-900">
                          {m.name}
                        </h3>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          {m.description}
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                      <span className="text-[11px] font-medium text-slate-500">{m.ctx}</span>
                      <div
                        className={`w-4 h-4 rounded-full border flex items-center justify-center ${
                          isSelected
                            ? "border-indigo-600 bg-indigo-600"
                            : "border-slate-300 bg-white"
                        }`}
                      >
                        {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* Bottom Action Footer */}
          <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="space-y-0.5 text-center sm:text-left">
              <p className="text-xs text-slate-500">
                Configuration Summary:
              </p>
              <p className="text-sm font-semibold text-slate-800">
                {selectedDomains.length ? selectedDomains.join(", ") : "No domain chosen"} •{" "}
                {selectedApproaches.length ? `${selectedApproaches.length} approach(es)` : "Standard approach"} •{" "}
                {primaryGoal} • {modelOptions.find(m => m.id === selectedModel)?.name}
              </p>
            </div>

            <div className="flex items-center space-x-3 w-full sm:w-auto justify-end">
              <Link
                href="/"
                className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-sm font-medium transition-colors"
              >
                Cancel
              </Link>

              {/* Create project and launch studio button */}
              <button
                type="button"
                onClick={handleCreateProject}
                disabled={isSubmitting}
                className="inline-flex items-center justify-center space-x-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm shadow-sm transition-all duration-150 active:scale-95 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-60"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Launching Studio...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Create project</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
