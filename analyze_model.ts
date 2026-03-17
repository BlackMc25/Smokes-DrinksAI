import fs from 'fs';

function analyzeModel(filename: string) {
  console.log(`\n--- Analyzing ${filename} ---`);
  const data = JSON.parse(fs.readFileSync(filename, 'utf8'));
  const featureNames = data.learner.feature_names;
  
  const featureCounts: Record<string, number> = {};
  featureNames.forEach((name: string) => featureCounts[name] = 0);

  data.learner.gradient_booster.model.trees.forEach((tree: any) => {
    if (tree.split_indices) {
      tree.split_indices.forEach((featIdx: number) => {
        if (featIdx !== -1 && featIdx < featureNames.length) {
          const name = featureNames[featIdx];
          featureCounts[name]++;
        }
      });
    }
  });

  console.log('Feature split counts (Importance):');
  Object.entries(featureCounts)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .forEach(([name, count]) => console.log(`${name}: ${count}`));
}

analyzeModel('public/Retrain_Drinking_model_xgb.json');
analyzeModel('public/Retrain_smoking_model_xgb.json');
