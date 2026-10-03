// Both demonstrations use synthetic two-dimensional data, not model activations.
export const DEFAULT_ANGLE = 60;
export const START = [-1.4, -1.2];
export const TARGET = [0.9, 1.2];
export const LEARNING_RATE = 0.25;
export const MAX_STEPS = 12;
export const VOCABULARY = [
  { word: 'summarize', vector: [-1.4, -1.2] },
  { word: 'classify', vector: [-1.15, 1] },
  { word: 'translate', vector: [0, 1.65] },
  { word: 'extract', vector: [1.5, 0.4] },
  { word: 'compare', vector: [0.25, -0.55] },
  { word: 'explain', vector: [1.3, -1.15] }
];

export function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map(score => Math.exp(score - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

export function attention(angle) {
  const radians = angle * Math.PI / 180;
  const key = [2 * Math.cos(radians), 2 * Math.sin(radians)];
  // q = [1, 0], k = [key, [0, 1], [-1, 0]], d_k = 2.
  const scores = [key[0] / Math.sqrt(2), 0, -1 / Math.sqrt(2)];
  const weights = softmax(scores);
  // v = [[1, 0], [0, 1], [-1, 0]]. All values stay fixed.
  const mixed = [weights[0] - weights[2], weights[1]];
  return { key, scores, weights, mixed };
}

export function loss(vector) {
  return vector.reduce((sum, value, i) => sum + (value - TARGET[i]) ** 2, 0) / 2;
}

export function gradientStep(vector) {
  return vector.map((value, i) => value - LEARNING_RATE * (value - TARGET[i]));
}

export function nearestToken(vector) {
  return VOCABULARY.reduce((best, token) => {
    const distance = token.vector.reduce((sum, value, i) => sum + (value - vector[i]) ** 2, 0);
    return distance < best.distance ? { ...token, distance } : best;
  }, { distance: Infinity });
}

export const plotPoint = ([x, y]) => [180 + 70 * x, 140 - 60 * y];
export const fixed = (value, digits = 2) => (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits);
export const formatVector = vector => `[${vector.map(value => fixed(value)).join(', ')}]`;
