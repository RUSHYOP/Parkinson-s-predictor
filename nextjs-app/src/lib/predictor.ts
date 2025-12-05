// XGBoost predictor using the exported JSON model
// This implements client-side inference without ONNX

import { VoiceFeatures, ScalerParams, ModelMeta, getRiskLevel } from './types';

interface XGBTree {
  nodeid: number;
  depth?: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: XGBTree[];
}

interface XGBModel {
  learner: {
    gradient_booster: {
      model: {
        trees: XGBTree[];
        tree_info: number[];
      };
    };
    learner_model_param: {
      base_score: string;
      num_class: string;
      num_feature: string;
    };
    objective: {
      name: string;
    };
  };
}

let modelCache: XGBModel | null = null;
let scalerCache: ScalerParams | null = null;
let metaCache: ModelMeta | null = null;

export async function loadModel(): Promise<{ model: XGBModel; scaler: ScalerParams; meta: ModelMeta }> {
  if (modelCache && scalerCache && metaCache) {
    return { model: modelCache, scaler: scalerCache, meta: metaCache };
  }

  const [modelRes, scalerRes, metaRes] = await Promise.all([
    fetch('/models/parkinsons_model.json'),
    fetch('/models/scaler.json'),
    fetch('/models/model_meta.json'),
  ]);

  if (!modelRes.ok || !scalerRes.ok || !metaRes.ok) {
    throw new Error('Failed to load model files');
  }

  modelCache = await modelRes.json();
  scalerCache = await scalerRes.json();
  metaCache = await metaRes.json();

  return { model: modelCache!, scaler: scalerCache!, meta: metaCache! };
}

function traverseTree(tree: XGBTree, features: number[], featureNames: string[]): number {
  // Base case: leaf node
  if (tree.leaf !== undefined) {
    return tree.leaf;
  }

  // Get feature index from feature name
  const featureIndex = featureNames.indexOf(tree.split!);
  if (featureIndex === -1) {
    console.error(`Feature ${tree.split} not found in feature names`);
    return 0;
  }

  const featureValue = features[featureIndex];
  
  // Find child nodes
  const children = tree.children || [];
  const yesNode = children.find(c => c.nodeid === tree.yes);
  const noNode = children.find(c => c.nodeid === tree.no);

  // Navigate based on split condition
  if (featureValue < (tree.split_condition || 0)) {
    return yesNode ? traverseTree(yesNode, features, featureNames) : 0;
  } else {
    return noNode ? traverseTree(noNode, features, featureNames) : 0;
  }
}

function predictTreeRecursive(node: XGBTree, features: number[], featureNames: string[]): number {
  // Leaf node
  if (node.leaf !== undefined) {
    return node.leaf;
  }

  // Internal node - find feature and compare
  const featureIndex = featureNames.indexOf(node.split!);
  const featureValue = featureIndex >= 0 ? features[featureIndex] : 0;

  // XGBoost uses "yes" for < split_condition
  if (featureValue < (node.split_condition || 0)) {
    // Go to "yes" child
    const yesChild = node.children?.find(c => c.nodeid === node.yes);
    return yesChild ? predictTreeRecursive(yesChild, features, featureNames) : 0;
  } else {
    // Go to "no" child  
    const noChild = node.children?.find(c => c.nodeid === node.no);
    return noChild ? predictTreeRecursive(noChild, features, featureNames) : 0;
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function scaleFeatures(features: VoiceFeatures, scaler: ScalerParams, featureOrder: string[]): number[] {
  const scaled: number[] = [];
  
  for (let i = 0; i < featureOrder.length; i++) {
    const featureName = featureOrder[i];
    const value = features[featureName as keyof VoiceFeatures] || 0;
    const mean = scaler.mean[i];
    const scale = scaler.scale[i];
    
    scaled.push((value - mean) / scale);
  }
  
  return scaled;
}

export async function predict(features: VoiceFeatures): Promise<{
  probability: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  riskPercentage: number;
  confidenceLower: number;
  confidenceUpper: number;
}> {
  const { model, scaler, meta } = await loadModel();
  
  // Scale features in the correct order
  const featureOrder = meta.feature_order || meta.feature_names;
  const scaledFeatures = scaleFeatures(features, scaler, featureOrder);
  
  // Get trees from model
  const trees = model.learner.gradient_booster.model.trees;
  const treeInfo = model.learner.gradient_booster.model.tree_info;
  
  // Parse base score
  let baseScore = 0;
  const baseScoreRaw = model.learner.learner_model_param.base_score;
  if (typeof baseScoreRaw === 'string') {
    baseScore = parseFloat(baseScoreRaw.replace(/[\[\]]/g, ''));
  } else {
    baseScore = baseScoreRaw as number;
  }
  
  // Accumulate predictions from all trees
  // For binary classification, we sum all tree outputs and apply sigmoid
  let sumPrediction = 0;
  
  for (let i = 0; i < trees.length; i++) {
    // For binary:logistic, all trees contribute to class 1 probability
    const treePrediction = predictTreeRecursive(trees[i], scaledFeatures, featureOrder);
    sumPrediction += treePrediction;
  }
  
  // Apply sigmoid to get probability of positive class (Parkinson's)
  const probability = sigmoid(sumPrediction);
  
  // Calculate confidence interval (simplified Wilson score interval)
  const z = 1.96; // 95% confidence
  const n = 100; // effective sample size approximation
  const p = probability;
  const denominator = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denominator;
  const spread = (z / denominator) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  
  const confidenceLower = Math.max(0, center - spread);
  const confidenceUpper = Math.min(1, center + spread);
  
  return {
    probability,
    riskLevel: getRiskLevel(probability),
    riskPercentage: probability * 100,
    confidenceLower: confidenceLower * 100,
    confidenceUpper: confidenceUpper * 100,
  };
}

export function isModelLoaded(): boolean {
  return modelCache !== null && scalerCache !== null && metaCache !== null;
}

export async function getModelInfo(): Promise<ModelMeta | null> {
  try {
    const { meta } = await loadModel();
    return meta;
  } catch {
    return null;
  }
}
