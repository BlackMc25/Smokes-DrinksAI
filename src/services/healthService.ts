
import { GoogleGenAI } from "@google/genai";

export interface HealthMetrics {
  hemoglobin: number;
  sex_numeric: number; // 1 for Male, 2 for Female (usually)
  SBP: number;
  DBP: number;
  BLDS: number;
  gamma_GTP: number;
  BMI: number;
  waistline: number;
  HDL_chole: number;
  LDL_chole: number;
  triglyceride: number;
  SMK_stat_type_cd: number; // 1: Never, 2: Ex-smoker, 3: Smoker
  DRK_YN: number; // 0: No, 1: Yes
  urine_protein: number;
}

export async function getHealthInsights(metrics: HealthMetrics, smokingProb: number, drinkingProb: number) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
  
  const prompt = `
    Based on the following health metrics and AI-predicted probabilities, provide a concise health assessment and actionable advice. You are Elena, a professional health assistant.
    
    Metrics:
    - BMI: ${metrics.BMI.toFixed(1)}
    - Blood Pressure: ${metrics.SBP}/${metrics.DBP} mmHg
    - Hemoglobin: ${metrics.hemoglobin} g/dL
    - Fasting Blood Sugar (BLDS): ${metrics.BLDS} mg/dL
    - Gamma-GTP: ${metrics.gamma_GTP} U/L
    - Waistline: ${metrics.waistline} cm
    - HDL Cholesterol: ${metrics.HDL_chole} mg/dL
    - LDL Cholesterol: ${metrics.LDL_chole} mg/dL
    - Triglycerides: ${metrics.triglyceride} mg/dL
    - Urine Protein: ${metrics.urine_protein}
    
    AI Predictions:
    - Probability of being a smoker: ${(smokingProb * 100).toFixed(1)}%
    - Probability of being a heavy drinker: ${(drinkingProb * 100).toFixed(1)}%
    
    Current Status:
    - Smoking Status: ${metrics.SMK_stat_type_cd === 1 ? 'Never' : metrics.SMK_stat_type_cd === 2 ? 'Ex-smoker' : 'Current Smoker'}
    - Drinking Status: ${metrics.DRK_YN === 1 ? 'Yes' : 'No'}
    
    Please provide:
    1. A summary of their cardiovascular and metabolic risk.
    2. Specific advice on how their metrics relate to smoking/drinking risks.
    3. 3-5 actionable steps to improve their vitality.
    
    Keep the tone professional yet encouraging. Use Markdown.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: prompt,
  });

  return response.text;
}
