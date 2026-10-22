"use client";

import dynamic from "next/dynamic";
import React from "react";

// Dynamically import Agentation with SSR disabled to ensure client-only execution in App Router
const Agentation = dynamic(
  () => import("agentation").then((mod) => mod.Agentation),
  { ssr: false }
);

export function AgentationDev() {
  if (process.env.NODE_ENV !== "development") {
    return null;
  }
  return <Agentation />;
}
