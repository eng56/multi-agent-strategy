export const dynamic = "force-dynamic";
export async function GET(_: Request, {params}: {params: Promise<{id:string}>}) {
  const {id}=await params;
  return fetch(`${process.env.ORCHESTRATOR_API_URL}/v1/runs/${id}/detail`, {headers:{"x-api-key":process.env.ORCHESTRATOR_API_TOKEN??""},cache:"no-store"});
}
