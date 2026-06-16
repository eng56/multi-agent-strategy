export const dynamic = "force-dynamic";
export async function POST(request: Request) {
  const body = await request.text();
  return fetch(`${process.env.ORCHESTRATOR_API_URL}/v1/runs`, {method:"POST",headers:{"content-type":"application/json","x-api-key":process.env.ORCHESTRATOR_API_TOKEN??""},body,cache:"no-store"});
}
