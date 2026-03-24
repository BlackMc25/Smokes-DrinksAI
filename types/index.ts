export interface HealthData {
  age: number;
  gender: 'male' | 'female';
  height: number;
  weight: number;
  waist: number;
  systolic: number;
  relaxation: number;
  fastingBloodSugar: number;
  cholesterol: number;
  triglyceride: number;
  hdl: number;
  ldl: number;
  hemoglobin: number;
  urineProtein: number;
  serumCreatinine: number;
  ast: number;
  alt: number;
  gtp: number;
  dentalCaries: boolean;
}

export interface PredictionResult {
  risk: number;
  level: 'Low' | 'Moderate' | 'High' | 'Critical';
  recommendations: string[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}
