import { GoogleGenAI, Modality } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export const chatModel = "gemini-3-flash-preview";
export const voiceModel = "gemini-2.5-flash-preview-tts";

export async function getChatResponse(message: string, history: { role: 'user' | 'model', parts: { text: string }[] }[]) {
  const chat = ai.chats.create({
    model: chatModel,
    config: {
      systemInstruction: "Your name is Elena. You are a health assistant specializing in smoking and drinking cessation and health monitoring. Provide empathetic, evidence-based advice. Avoid medical diagnoses but offer risk assessment and motivational support.",
    },
  });

  // We don't actually pass history to ai.chats.create in the SDK example provided, 
  // but we can use sendMessage. For simplicity in this turn, we'll just use generateContent for now or manage history manually if needed.
  // Actually, the SDK says ai.chats.create.
  
  const response = await ai.models.generateContent({
    model: chatModel,
    contents: [...history, { role: 'user', parts: [{ text: message }] }],
    config: {
      systemInstruction: "Your name is Elena. You are a health assistant specializing in smoking and drinking cessation and health monitoring.",
    }
  });

  return response.text;
}

export async function getVoiceResponse(text: string) {
  const response = await ai.models.generateContent({
    model: voiceModel,
    contents: [{ parts: [{ text: `Say clearly and supportively: ${text}` }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: 'Kore' },
        },
      },
    },
  });

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  return base64Audio;
}

export async function getHealthPrediction(data: { smokingFrequency: string, drinkingFrequency: string, age: number, duration: string }) {
  const prompt = `Based on the following data, provide a health risk prediction and motivational advice:
  Age: ${data.age}
  Smoking: ${data.smokingFrequency}
  Drinking: ${data.drinkingFrequency}
  Duration of habits: ${data.duration}
  
  Provide the response in JSON format with keys: "riskLevel" (Low, Moderate, High, Critical), "prediction" (short summary), "advice" (3 bullet points).`;

  const response = await ai.models.generateContent({
    model: chatModel,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
    }
  });

  return JSON.parse(response.text || "{}");
}
