export interface XGBoostTree {
  left_children: number[];
  right_children: number[];
  split_indices: number[];
  split_conditions: number[];
  base_weights: number[];
  default_left: number[];
}

export interface XGBoostModel {
  learner: {
    learner_model_param?: {
      base_score?: string;
      num_class?: string;
    };
    gradient_booster: {
      model: {
        trees: XGBoostTree[];
      };
    };
  };
}

export function predictXGBoostMulticlass(model: XGBoostModel, features: number[]): number[] {
  const trees = model.learner.gradient_booster.model.trees;
  const numClass = parseInt(model.learner.learner_model_param?.num_class || "1");
  
  let baseScores: number[] = [];
  if (model.learner.learner_model_param?.base_score) {
    try {
      const scoreStr = model.learner.learner_model_param.base_score;
      // It might be "[val1,val2,...]" or just a number
      if (scoreStr.startsWith("[")) {
        baseScores = JSON.parse(scoreStr);
      } else {
        const val = parseFloat(scoreStr);
        baseScores = new Array(numClass).fill(val);
      }
    } catch (e) {
      console.error("Failed to parse base_score", e);
      baseScores = new Array(numClass).fill(0.5);
    }
  } else {
    baseScores = new Array(numClass).fill(0.5);
  }

  const logits = [...baseScores];

  for (let i = 0; i < trees.length; i++) {
    const classIdx = i % numClass;
    const tree = trees[i];
    
    let nodeIdx = 0;
    while (tree.left_children[nodeIdx] !== -1) {
      const featureIdx = tree.split_indices[nodeIdx];
      const condition = tree.split_conditions[nodeIdx];
      const val = features[featureIdx];

      // XGBoost default behavior for missing values
      if (val === undefined || val === null || isNaN(val)) {
        if (tree.default_left[nodeIdx]) {
          nodeIdx = tree.left_children[nodeIdx];
        } else {
          nodeIdx = tree.right_children[nodeIdx];
        }
      } else if (val < condition) {
        nodeIdx = tree.left_children[nodeIdx];
      } else {
        nodeIdx = tree.right_children[nodeIdx];
      }
    }
    logits[classIdx] += tree.base_weights[nodeIdx];
  }

  if (numClass === 1) {
    // Binary classification
    const prob = 1 / (1 + Math.exp(-logits[0]));
    return [1 - prob, prob];
  } else {
    // Softmax
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }
}
