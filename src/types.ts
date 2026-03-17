export interface HealthData {
  sex: 'Male' | 'Female';
  age: number;
  height: number;
  weight: number;
  waistline: number;
  SBP: number;
  DBP: number;
  BLDS: number;
  tot_chole: number;
  HDL_chole: number;
  LDL_chole: number;
  triglyceride: number;
  hemoglobin: number;
  urine_protein: number;
  serum_creatinine: number;
  SGOT_AST: number;
  SGOT_ALT: number;
  gamma_GTP: number;
  // Status features for cross-prediction
  is_drinker: number; // 0 or 1
  smoking_status: number; // 0, 1, or 2
}

export interface PredictionResult {
  smokingProbs: number[];
  drinkingProbs: number[];
}
