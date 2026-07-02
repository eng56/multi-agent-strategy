export const dynamic = "force-dynamic";
export async function GET() {
  return fetch(`${process.env.ORCHESTRATOR_API_URL}/v1/models/openrouter`, {headers:{"x-api-key":process.env.ORCHESTRATOR_API_TOKEN??""},cache:"no-store"});
}
