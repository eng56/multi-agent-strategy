import {NextRequest, NextResponse} from "next/server";
export function middleware(request: NextRequest) {
  const {pathname}=request.nextUrl;
  if (pathname.startsWith("/login") || pathname.startsWith("/api/login") || pathname.startsWith("/_next") || pathname === "/favicon.ico") return NextResponse.next();
  if (request.cookies.get("demo_auth")?.value === "1") return NextResponse.next();
  return NextResponse.redirect(new URL("/login", request.url));
}
export const config = {matcher: ["/((?!_next/static|_next/image).*)"]};
