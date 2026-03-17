
export interface XGBModel {
  learner: {
    feature_names: string[];
    gradient_booster: {
      model: {
        trees: {
          left_children: number[];
          right_children: number[];
          split_indices: number[];
          split_conditions: number[];
          base_weights: number[];
        }[];
        tree_info?: number[];
      };
    };
    learner_model_param?: {
      base_score?: string | number;
      num_class?: string | number;
    };
  };
}

export class XGBPredictor {
  private model: XGBModel | null = null;

  constructor(modelJson: XGBModel) {
    this.model = modelJson;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(s => s / sumExp);
  }

  predict(features: Record<string, number>): number {
    if (!this.model) return 0;

    const featureNames = this.model.learner.feature_names;
    const trees = this.model.learner.gradient_booster.model.trees;
    const treeInfo = this.model.learner.gradient_booster.model.tree_info;
    const modelParam = this.model.learner.learner_model_param;
    
    const numClass = modelParam?.num_class ? parseInt(modelParam.num_class.toString()) : 1;
    let baseScores: number[] = [];

    if (modelParam?.base_score) {
      const bs = modelParam.base_score.toString();
      if (bs.startsWith('[')) {
        try {
          baseScores = JSON.parse(bs);
        } catch (e) {
          baseScores = [parseFloat(bs) || 0.5];
        }
      } else {
        baseScores = [parseFloat(bs)];
      }
    } else {
      baseScores = new Array(numClass).fill(0.5);
    }

    // If it's a multi-class model, we need to sum trees for each class
    if (numClass > 1) {
      const classScores = new Array(numClass).fill(0);
      
      for (let i = 0; i < trees.length; i++) {
        const tree = trees[i];
        const classIdx = treeInfo ? treeInfo[i] : (i % numClass);
        
        let nodeIdx = 0;
        while (tree.left_children[nodeIdx] !== -1) {
          const featureIdx = tree.split_indices[nodeIdx];
          const featureName = featureNames[featureIdx];
          const threshold = tree.split_conditions[nodeIdx];
          const value = features[featureName] ?? 0;

          if (value < threshold) {
            nodeIdx = tree.left_children[nodeIdx];
          } else {
            nodeIdx = tree.right_children[nodeIdx];
          }
        }
        classScores[classIdx] += tree.base_weights[nodeIdx];
      }

      // Add base scores
      for (let i = 0; i < numClass; i++) {
        classScores[i] += baseScores[i] || 0;
      }

      const probs = this.softmax(classScores);
      // For risk, we usually want the probability of non-zero classes
      // Assuming class 0 is "normal/no-risk"
      return 1 - probs[0];
    } else {
      // Binary classification or regression
      let score = 0;
      for (const tree of trees) {
        let nodeIdx = 0;
        while (tree.left_children[nodeIdx] !== -1) {
          const featureIdx = tree.split_indices[nodeIdx];
          const featureName = featureNames[featureIdx];
          const threshold = tree.split_conditions[nodeIdx];
          const value = features[featureName] ?? 0;

          if (value < threshold) {
            nodeIdx = tree.left_children[nodeIdx];
          } else {
            nodeIdx = tree.right_children[nodeIdx];
          }
        }
        score += tree.base_weights[nodeIdx];
      }
      score += baseScores[0] || 0;
      return this.sigmoid(score);
    }
  }
}

export async function loadModel(url: string): Promise<XGBPredictor> {
  const response = await fetch(url);
  const json = await response.json();
  return new XGBPredictor(json);
}
