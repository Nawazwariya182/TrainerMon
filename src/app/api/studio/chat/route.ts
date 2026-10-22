import { NextRequest, NextResponse } from "next/server";

const API_KEY = "sk-mltrain-live-engine-98f2b4c107e3a9d2";
const INFERENCE_URL = "http://127.0.0.1:8000/v1/chat/completions";
const RAG_QUERY_URL = "http://127.0.0.1:8000/v1/research/rag/query";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      messages = [],
      model = "qwen3-8b",
      projectContext = null,
    } = body;

    // Retrieve semantically relevant chunks from ChromaDB RAG
    let ragContextText = "";
    try {
      const lastUserMsg = [...messages].reverse().find((m: any) => m.role === "user")?.content || "";
      const projectId = projectContext?.project?.id;
      if (lastUserMsg) {
        const ragRes = await fetch(RAG_QUERY_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${API_KEY}`,
          },
          body: JSON.stringify({
            project_id: projectId,
            query: lastUserMsg,
            top_k: 4,
          }),
          signal: AbortSignal.timeout(3000),
        });
        if (ragRes.ok) {
          const ragData = await ragRes.json();
          if (Array.isArray(ragData.results) && ragData.results.length > 0) {
            ragContextText =
              "\n\nRETRIEVED KNOWLEDGE BASE CONTEXT (ChromaDB RAG):\n" +
              ragData.results
                .map(
                  (r: any, idx: number) =>
                    `[Doc ${idx + 1} - ${r.metadata?.title || r.metadata?.doc_type || "Context"}]:\n${r.content}`
                )
                .join("\n\n");
          }
        }
      }
    } catch {
      // Graceful fallback if RAG query times out
    }

    const systemPrompt = `You are the ML-Train Studio AI Assistant, an expert Machine Learning and Deep Learning Systems Architect.
IMPORTANT DIRECTIVES:
1. EXCLUSIVELY TRAINING MODELS FROM SCRATCH: This studio is dedicated strictly to training custom neural network architectures and models from scratch (from randomly initialized weights, custom feature/token representations, loss formulations, and optimizer schedules). DO NOT suggest or reference fine-tuning, LoRA, QLoRA, or pretrained checkpoints.
2. DOMAIN & MODALITY AWARENESS:
   Analyze the loaded project context. Ground your answers specifically in the user's modality:
   - If Computer Vision / Medical Imaging (e.g. Brain Tumor MRI): Discuss 3D volumetric tensors, 3D U-Net, Swin UNETR, Soft Dice Loss + Focal Loss, skull-stripping/N4 bias correction, 3D elastic augmentations, random He/Kaiming initialization, and sliding-window inference.
   - If Audio / Speech: Discuss spectrograms, raw waveforms, Conformer backbones, CTC / RNN-T loss, and SpecAugment.
   - If Time Series: Discuss lookback windows, forecasting horizons, PatchTST / TimesNet, RevIN, and quantile loss.
   - If Tabular: Discuss GBDT, TabNet, target encoding, and Out-Of-Fold validation.
   - If Text / NLP: Discuss tokenizers, vocab size, transformer decoders, scaling laws, and next-token prediction.
3. HANDLING IMPOSSIBLE / INFEASIBLE REQUESTS: If the user asks for something physically or statistically impossible (such as predicting lottery numbers, 100% stock market certainty, or pretraining a massive model without compute/data), reply starting with "Sorry...", explain the exact scientific limitations, and provide viable alternative ways to train from scratch.
4. ASSISTING WITH RESEARCH PLAN EDITS: If the user asks to modify the technical research (e.g. change the optimizer, adjust loss weights, replace the architecture, or add new TODOs), acknowledge their preference with engineering rationale, explain the exact adjustments, and present the updated values clearly.
5. Keep answers concise, highly technical, actionable, and encouraging.

Project Context:
${projectContext ? JSON.stringify(projectContext, null, 2) : "No specific project loaded."}
${ragContextText}
`;

    const formattedMessages = [
      { role: "system", content: systemPrompt },
      ...messages,
    ];

    const response = await fetch(INFERENCE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: model || "qwen3-8b",
        messages: formattedMessages,
        max_tokens: 1024,
        temperature: 0.5,
        stop: ["<|im_end|>", "<|endoftext|>"],
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Inference server returned status ${response.status}`);
    }

    const data = await response.json();
    const rawContent: string = data.choices?.[0]?.message?.content || "";

    // Extract <think> trace if present
    let thinkingTrace = "";
    let cleanContent = rawContent;
    const thinkMatch = rawContent.match(/<think>([\s\S]*?)<\/think>/i);
    if (thinkMatch) {
      thinkingTrace = thinkMatch[1].trim();
      cleanContent = rawContent.replace(/<think>[\s\S]*?<\/think>/i, "").trim();
    } else if (rawContent.includes("<think>")) {
      const parts = rawContent.split("<think>");
      if (parts[1]) {
        thinkingTrace = parts[1].replace(/<\/think>[\s\S]*/, "").trim();
        cleanContent = parts[0].trim();
        if (!cleanContent) {
          cleanContent = thinkingTrace;
        }
      }
    }

    return NextResponse.json({
      success: true,
      message: {
        role: "assistant",
        content: cleanContent || "I have analyzed your research and dataset specifications. How would you like to proceed?",
        thinking: thinkingTrace,
      },
      model_used: model,
      rag_augmented: !!ragContextText,
      metrics: data.inference_metrics,
    });
  } catch (err: any) {
    console.error("Studio chat error:", err);
    return NextResponse.json(
      { success: false, error: err.message || "Failed to generate chat response" },
      { status: 500 }
    );
  }
}
