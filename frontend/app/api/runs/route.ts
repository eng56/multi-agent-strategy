export const dynamic = "force-dynamic";
export const maxDuration = 60;

function jsonError(detail: string, status: number) {
  return Response.json({detail}, {status});
}

function truncate(value: string) {
  return value.length > 500 ? `${value.slice(0, 500)}…` : value;
}

export async function POST(request: Request) {
  const apiUrl = process.env.ORCHESTRATOR_API_URL;
  const apiToken = process.env.ORCHESTRATOR_API_TOKEN;
  if (!apiUrl || !apiToken) {
    return jsonError("Frontend is missing ORCHESTRATOR_API_URL or ORCHESTRATOR_API_TOKEN", 500);
  }
  const body = await request.text();
  try {
    const response = await fetch(`${apiUrl}/v1/runs`, {
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
    return jsonError(timedOut ? "Backend provider validation timed out" : "Could not reach orchestrator API", timedOut ? 504 : 502);
  }
}
