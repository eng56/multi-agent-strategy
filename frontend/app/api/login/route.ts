export const dynamic = "force-dynamic";
export async function POST(request: Request) {
  const {password}=await request.json();
  if (!process.env.DEMO_PASSWORD || password !== process.env.DEMO_PASSWORD) {
    return Response.json({error:"invalid password"},{status:401});
  }
  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  return new Response(null,{status:204,headers:{"set-cookie":`demo_auth=1; Path=/; HttpOnly${secure}; SameSite=Lax; Max-Age=86400`}});
}
