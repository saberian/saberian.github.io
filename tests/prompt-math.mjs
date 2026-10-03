import assert from 'node:assert/strict';
import test from 'node:test';
import { attention, softmax, START, TARGET, MAX_STEPS, VOCABULARY, gradientStep, loss, nearestToken, plotPoint } from '../assets/prompt-math.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} ≈ ${expected}`);

test('attention is the scaled dot product, normalized softmax, and weighted value sum', () => {
  let previous = Infinity;
  for (let angle = 0; angle <= 180; angle += 5) {
    const { key, scores, weights, mixed } = attention(angle);
    close(Math.hypot(...key), 2);
    close(scores[0], key[0] / Math.sqrt(2));
    close(weights.reduce((a, b) => a + b), 1);
    assert.ok(weights.every(weight => weight > 0 && weight < 1));
    assert.ok(weights[0] < previous);
    previous = weights[0];
    close(mixed[0], weights[0] - weights[2]);
    close(mixed[1], weights[1]);
  }
  close(attention(60).weights[0], 0.575975345215362);
  close(attention(90).weights[0], attention(90).weights[1]);
});

test('softmax remains finite for large scores', () => {
  assert.deepEqual(softmax([1000, 1000]), [0.5, 0.5]);
  assert.deepEqual(softmax([-1000, -1000]), [0.5, 0.5]);
});

test('every gradient step reduces the declared toy loss without changing the vocabulary', () => {
  let vector = [...START];
  const vocabulary = JSON.stringify(VOCABULARY);
  for (let step = 0; step < MAX_STEPS; step++) {
    const next = gradientStep(vector);
    close(loss(next), loss(vector) * 0.75 ** 2);
    assert.ok(loss(next) < loss(vector));
    vector = next;
  }
  assert.ok(loss(vector) < 0.006);
  close(loss(TARGET), 0);
  assert.equal(JSON.stringify(VOCABULARY), vocabulary);
  assert.equal(nearestToken(vector).word, 'translate');
  assert.equal(nearestToken(TARGET).word, 'extract');
  assert.ok(loss(nearestToken(vector).vector) > loss(vector));
});

test('nearest-token snapping and chart coordinates agree with the static illustration', () => {
  for (const token of VOCABULARY) assert.equal(nearestToken(token.vector).word, token.word);
  assert.deepEqual(plotPoint(START), [82, 212]);
  assert.deepEqual(plotPoint(TARGET), [243, 68]);
});
