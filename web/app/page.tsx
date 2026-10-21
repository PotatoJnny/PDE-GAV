import { headers } from "next/headers";

import {
  chatGPTSignOutPath,
  requireChatGPTUser,
} from "./chatgpt-auth";
import Dashboard from "./dashboard";

export const dynamic = "force-dynamic";

export default async function Home() {
  const requestHeaders = await headers();
  const hostname = (requestHeaders.get("host") ?? "").split(":")[0];
  const isLocal = hostname === "localhost" || hostname === "127.0.0.1";
  const viewerName = isLocal
    ? "Local preview"
    : (await requireChatGPTUser("/")).displayName;

  return (
    <Dashboard
      viewerName={viewerName}
      signOutPath={isLocal ? "" : chatGPTSignOutPath("/")}
    />
  );
}
