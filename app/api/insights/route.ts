import { GoogleGenAI } from "@google/genai";

export async function POST(req: Request) {
  try {
    const { smoking, drinking } = await req.json();

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY!,
    });

    const MODEL_PRIMARY = "gemini-2.5-flash";
    const MODEL_FALLBACK = "gemini-1.5-flash";

    const prompt = `
As a health AI, provide a brief (2-3 sentences) insight for a patient with:
- Smoking risk: ${(smoking * 100).toFixed(1)}%
- Drinking risk: ${(drinking * 100).toFixed(1)}%

Be encouraging, professional, and natural.
Avoid repetition and give meaningful interpretation.
`;

    let response;

    try {
      response = await ai.models.generateContent({
        model: MODEL_PRIMARY,
        contents: prompt,
        config: {
          temperature: 0.9,
          topP: 0.95,
          topK: 64,
        },
      });
    } catch (primaryError) {
      console.warn("Primary model failed, switching fallback...");

      response = await ai.models.generateContent({
        model: MODEL_FALLBACK,
        contents: prompt,
        config: {
          temperature: 0.9,
          topP: 0.95,
          topK: 64,
        },
      });
    }

    return Response.json({
      text: response.text || "No insight generated",
    });

  } catch (error) {
    console.error("INSIGHT API ERROR:", error);

    return new Response(
      JSON.stringify({
        error: "AI temporarily unavailable. Please try again."
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}