import { attention, DEFAULT_ANGLE, START, TARGET, MAX_STEPS, gradientStep, loss, nearestToken, plotPoint, formatVector, fixed } from './prompt-math.js';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function enable(figure) {
  figure.querySelector('[data-controls]').hidden = false;
  figure.querySelector('[data-fallback]').hidden = true;
}

function initAttention(figure) {
  const slider = figure.querySelector('input');
  const rows = [...figure.querySelectorAll('[data-score-row]')];
  function render() {
    const angle = Number(slider.value);
    const { key, scores, weights, mixed } = attention(angle);
    figure.querySelector('[data-key-arm]').style.transform = `rotate(${-angle}deg)`;
    figure.querySelector('[data-angle]').textContent = `${angle}°`;
    rows.forEach((row, index) => {
      row.querySelector('[data-score]').textContent = fixed(scores[index], 3);
      row.querySelector('[data-weight]').textContent = `${fixed(weights[index] * 100, 1)}%`;
      row.querySelector('[data-bar]').style.transform = `scaleX(${weights[index]})`;
    });
    figure.querySelector('[data-mixed]').textContent = formatVector(mixed);
    figure.querySelector('[data-key]').textContent = `Prompt key = ${formatVector(key)}`;
    figure.querySelector('#attention-chart-desc').textContent = `The query points right. The prompt key is at ${angle} degrees and receives ${fixed(weights[0] * 100, 1)} percent of the attention. The mixed value is ${formatVector(mixed)}.`;
  }
  slider.addEventListener('input', render);
  figure.querySelector('[data-attention-reset]').addEventListener('click', () => {
    slider.value = DEFAULT_ANGLE;
    render();
  });
  render();
  enable(figure);
}

function initEmbedding(figure) {
  let vector = [...START];
  let history = [vector];
  let steps = 0;
  let timer = null;
  const stepButton = figure.querySelector('[data-step]');
  const playButton = figure.querySelector('[data-play]');
  const snap = figure.querySelector('[data-snap]');
  const point = figure.querySelector('[data-point]');

  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    playButton.textContent = reducedMotion.matches ? 'Finish steps' : 'Animate steps';
    playButton.setAttribute('aria-pressed', 'false');
  }

  function render() {
    const token = nearestToken(vector);
    const displayed = snap.checked ? token.vector : vector;
    const [x, y] = plotPoint(displayed);
    const [latentX, latentY] = plotPoint(vector);
    point.style.transform = `translate(${x}px, ${y}px)`;
    const ghost = figure.querySelector('[data-ghost]');
    ghost.style.transform = `translate(${latentX}px, ${latentY}px)`;
    // SVGElement has no HTML hidden property; set the attribute explicitly.
    ghost.toggleAttribute('hidden', !snap.checked);
    const snapPath = figure.querySelector('[data-snap-path]');
    snapPath.toggleAttribute('hidden', !snap.checked);
    snapPath.setAttribute('d', `M${latentX} ${latentY}L${x} ${y}`);
    figure.querySelector('[data-path]').setAttribute('d', history.map((p, i) => `${i ? 'L' : 'M'}${plotPoint(p).join(' ')}`).join(' '));
    figure.querySelector('[data-step-label]').textContent = `Step ${steps} of ${MAX_STEPS}`;
    figure.querySelector('[data-position]').textContent = formatVector(displayed);
    const snapHint = steps === MAX_STEPS
      ? 'Run complete. Uncheck to compare; reset to start again.'
      : 'Uncheck to continue learning.';
    const status = snap.checked
      ? `Nearest token: “${token.word}” · toy loss ${fixed(loss(displayed), 3)} (continuous: ${fixed(loss(vector), 3)}). ${snapHint}`
      : `Continuous vector · toy loss ${fixed(loss(vector), 3)}${steps === MAX_STEPS ? ' · run complete' : ''}`;
    figure.querySelector('[data-embedding-status]').textContent = status;
    figure.querySelector('#embedding-chart-desc').textContent = `Step ${steps}. ${snap.checked ? `Snapped to ${token.word}` : 'Continuous prompt'} at ${formatVector(displayed)}. Target ${formatVector(TARGET)}. ${status}`;
    stepButton.disabled = snap.checked || steps >= MAX_STEPS;
    playButton.disabled = snap.checked || steps >= MAX_STEPS;
  }

  function step() {
    if (steps >= MAX_STEPS || snap.checked) return;
    vector = gradientStep(vector);
    history.push(vector);
    steps += 1;
    if (steps === MAX_STEPS) stop();
    render();
  }

  stepButton.addEventListener('click', () => { stop(); step(); });
  playButton.addEventListener('click', () => {
    if (timer !== null) { stop(); return; }
    if (reducedMotion.matches) {
      while (steps < MAX_STEPS) { vector = gradientStep(vector); history.push(vector); steps += 1; }
      render();
      return;
    }
    playButton.textContent = 'Pause';
    playButton.setAttribute('aria-pressed', 'true');
    timer = window.setInterval(step, 400);
  });
  figure.querySelector('[data-reset]').addEventListener('click', () => {
    stop(); vector = [...START]; history = [vector]; steps = 0; snap.checked = false; render();
  });
  snap.addEventListener('change', () => { stop(); render(); });
  reducedMotion.addEventListener('change', stop);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) stop();
    });
    observer.observe(figure);
  }
  stop(); render(); enable(figure);
}

document.querySelectorAll('[data-attention]').forEach(initAttention);
document.querySelectorAll('[data-embedding]').forEach(initEmbedding);
