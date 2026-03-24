import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Elena - AI Health Companion",
  description: "Predict and understand your health risks with Elena.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
