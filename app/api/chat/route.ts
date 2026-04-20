import { GoogleGenAI } from "@google/genai";

export async function POST(req: Request) {
  try {
    const { message, history, image } = await req.json();

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY!,
    });

    const systemPrompt = `
You are Elena, a supportive health AI assistant.
Focus only on smoking, drinking, and wellness.
Be natural, helpful, and non-repetitive.
`;

    let response;

    const MODEL_PRIMARY = "gemini-2.5-flash";
    const MODEL_FALLBACK = "gemini-2.5-flash-lite";

    try {
      if (image) {
        const base64Data = image.split(",")[1];
        const mimeType = image.split(",")[0].split(":")[1].split(";")[0];

        response = await ai.models.generateContent({
          model: MODEL_PRIMARY,
          contents: [
            {
              role: "user",
              parts: [
                { text: systemPrompt },
                { text: message || "Analyze this image" },
                { inlineData: { data: base64Data, mimeType } }
              ]
            }
          ],
        });
      } else {
        const chat = ai.chats.create({
          model: MODEL_PRIMARY,
          history: [
            {
              role: "user",
              parts: [{ text: systemPrompt }]
            },
            ...(history || [])
          ],
        });

        response = await chat.sendMessage({ message });
      }
    } catch (err: any) {
      // 🔁 fallback model if rate-limited / not found / overloaded
      const chat = ai.chats.create({
        model: MODEL_FALLBACK,
        history: history || [],
      });

      response = await chat.sendMessage({ message });
    }

    return Response.json({
      text: response.text || "No response"
    });

  } catch (error) {
    console.error("API ERROR:", error);

    return new Response(
      JSON.stringify({
        error: "AI temporarily unavailable. Please try again."
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" }
      }
    );
  }
}