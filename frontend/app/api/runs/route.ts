export const dynamic = "force-dynamic";
export const maxDuration = 60;

function jsonError(detail: string, status: number) {
  return Response.json({detail}, {status});
}

function truncate(value: string) {
  return value.length > 500 ? `${value.slice(0, 500)}…` : value;
}

function orchestratorBaseUrl(value: string) {
  const url = new URL(value);
  url.pathname = url.pathname.replace(/\/(healthz|v1\/runs)\/?$/, "");
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

export async function POST(request: Request) {
  const apiUrl = process.env.ORCHESTRATOR_API_URL;
  const apiToken = process.env.ORCHESTRATOR_API_TOKEN;
  if (!apiUrl || !apiToken) {
    return jsonError("Frontend is missing ORCHESTRATOR_API_URL or ORCHESTRATOR_API_TOKEN", 500);
  }
  const body = await request.text();
  try {
    const baseUrl = orchestratorBaseUrl(apiUrl);
    const response = await fetch(`${baseUrl}/v1/runs`, {
      method: "POST",
      headers: {"content-type": "application/json", "x-api-key": apiToken},
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(55000),
    });
    const text = await response.text();
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      return new Response(text, {status: response.status, headers: {"content-type": "application/json"}});
    }
    return jsonError(
      `Orchestrator returned non-JSON response (${response.status} ${response.statusText}): ${truncate(text)}`,
      response.status,
    );
  } catch (exc) {
    const timedOut = exc instanceof DOMException && exc.name === "TimeoutError";
    const invalidUrl = exc instanceof TypeError;
    return jsonError(
      timedOut
        ? "Backend provider validation timed out"
        : invalidUrl
          ? "ORCHESTRATOR_API_URL must be the orchestrator base URL, for example http://34.27.225.48"
          : "Could not reach orchestrator API",
      timedOut ? 504 : 502,
    );
  }
}
