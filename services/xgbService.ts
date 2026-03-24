import { HealthData, PredictionResult } from '../types';

export interface XGBPredictor {
  predict(features: Record<string, number>): number;
}

class XGBModel implements XGBPredictor {
  private trees: any[];
  private featureNames: string[];
  private baseScore: number;

  constructor(modelData: any) {
    // XGBoost JSON format can vary slightly depending on version
    const learner = modelData.learner;
    this.featureNames = learner.feature_names;
    
    // Handle potential differences in base_score location
    const modelParam = learner.learner_model_param || learner.attributes;
    this.baseScore = parseFloat(modelParam?.base_score || "0.5");
    
    this.trees = learner.gradient_booster.model.trees;
  }

  predict(features: Record<string, number>): number {
    let score = 0;
    for (const tree of this.trees) {
      score += this.predictTree(tree, features);
    }
    
    // Add base score
    const totalScore = this.baseScore + score;
    
    // Sigmoid function for binary classification (probability)
    return 1 / (1 + Math.exp(-totalScore));
  }

  private predictTree(tree: any, features: Record<string, number>): number {
    let nodeIdx = 0;
    
    // Tree structure in XGBoost JSON:
    // left_children, right_children, split_indices, split_conditions, base_weights
    
    while (true) {
      const leftChild = tree.left_children[nodeIdx];
      const rightChild = tree.right_children[nodeIdx];

      // If both children are -1, it's a leaf node
      if (leftChild === -1 && rightChild === -1) {
        return tree.base_weights[nodeIdx];
      }

      const featureIdx = tree.split_indices[nodeIdx];
      const featureName = this.featureNames[featureIdx];
      const featureValue = features[featureName] ?? 0; // Default to 0 if missing
      const splitCondition = tree.split_conditions[nodeIdx];

      // XGBoost split: if feature < condition go left, else go right
      if (featureValue < splitCondition) {
        nodeIdx = leftChild;
      } else {
        nodeIdx = rightChild;
      }
      
      // Safety break for malformed trees
      if (nodeIdx === -1) return 0;
    }
  }
}

export async function loadModel(url: string): Promise<XGBPredictor> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load model from ${url}`);
  }
  const modelData = await response.json();
  return new XGBModel(modelData);
}

export class XGBService {
  static async predictSmokingRisk(data: HealthData): Promise<PredictionResult> {
    // Simulated XGBoost logic based on common health indicators
    let score = 0;
    
    if (data.hemoglobin > 15) score += 0.2;
    if (data.gtp > 40) score += 0.15;
    if (data.triglyceride > 150) score += 0.1;
    if (data.waist > 90) score += 0.1;
    if (data.systolic > 130) score += 0.1;
    if (data.dentalCaries) score += 0.15;

    const risk = Math.min(score, 1);
    return this.mapToResult(risk, 'Smoking');
  }

  static async predictDrinkingRisk(data: HealthData): Promise<PredictionResult> {
    let score = 0;
    
    if (data.alt > 35) score += 0.25;
    if (data.ast > 35) score += 0.2;
    if (data.gtp > 50) score += 0.25;
    if (data.triglyceride > 160) score += 0.15;
    if (data.hdl < 40) score += 0.1;

    const risk = Math.min(score, 1);
    return this.mapToResult(risk, 'Drinking');
  }

  private static mapToResult(risk: number, type: string): PredictionResult {
    let level: 'Low' | 'Moderate' | 'High' | 'Critical' = 'Low';
    let recommendations: string[] = [];

    if (risk < 0.3) {
      level = 'Low';
      recommendations = [`Your ${type.toLowerCase()} related health markers look good. Maintain your current lifestyle.`];
    } else if (risk < 0.6) {
      level = 'Moderate';
      recommendations = [`Consider reducing ${type.toLowerCase()} frequency.`, 'Increase physical activity.', 'Monitor liver enzymes regularly.'];
    } else if (risk < 0.8) {
      level = 'High';
      recommendations = [`Strongly advised to quit ${type.toLowerCase()}.`, 'Consult a healthcare professional.', 'Schedule a full metabolic panel.'];
    } else {
      level = 'Critical';
      recommendations = [`Immediate medical consultation required regarding ${type.toLowerCase()} habits.`, 'High risk of chronic conditions detected.'];
    }

    return { risk, level, recommendations };
  }
}
