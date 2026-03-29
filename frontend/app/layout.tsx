import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GroceryCast",
  description: "AI-powered forecasting of Canadian grocery prices."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
