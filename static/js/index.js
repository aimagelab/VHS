window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function () { return false; };
  image.oncontextmenu = function () { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function () {
  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Initialize carousel with proper options
  var carousels = bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 3000,
    navigation: true,
    navigationKeys: true,
    pagination: true
  });

  // Initialize all sliders
  bulmaSlider.attach();

  /*var player = document.getElementById('interpolation-video');
  player.addEventListener('loadedmetadata', function() {
    $('#interpolation-slider').on('input', function(event) {
      console.log(this.value, player.duration);
      player.currentTime = player.duration / 100 * this.value;
    })
  }, false);*/
  preloadInterpolationImages();

  $('#interpolation-slider').on('input', function (event) {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
});

document.addEventListener('DOMContentLoaded', function () {
  // Initialize all carousels
  var carousels = bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 3000,
    navigation: true
  });

  // Initialize all sliders
  bulmaSlider.attach();
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

window.addEventListener("load", () => {
  const steps = document.querySelectorAll(".step");

  steps.forEach((el, i) => {
    setTimeout(() => {
      el.style.opacity = 1;
      el.style.transform = "translateY(0)";
      el.style.transition = "all 0.5s ease";
    }, i * 300);
  });
});

// function animateMethod() {
//   const layers = document.querySelectorAll("#method .layer");
//   const text = document.querySelector(".cut-text");
//   const line = document.querySelector(".cut-line");

//   layers.forEach((el, i) => {
//     setTimeout(() => {
//       el.style.opacity = 1;
//       el.style.transform = "translateY(0)";
//       el.style.transition = "all 0.4s ease";
//     }, i * 200);
//   });

//   setTimeout(() => {
//     text.style.opacity = 1;
//     text.style.transition = "opacity 0.6s";
//   }, 1500);

//   setTimeout(() => {
//     line.style.width = "60%";
//     line.style.transition = "width 0.8s";
//   }, 1700);
// }

// window.addEventListener("load", animateMethod);

function animateProblem() {
  const generation = document.querySelector(".generation");
  const decode = document.querySelector(".decode");
  const encode = document.querySelector(".encode");
  const verify = document.querySelector(".verify");
  const text = document.querySelector(".waste-text");

  setTimeout(() => {
    generation.style.width = "30%";
    generation.style.transition = "width 0.8s";
  }, 0);

  setTimeout(() => {
    decode.style.width = "25%";
    decode.style.transition = "width 0.8s";
  }, 400);

  setTimeout(() => {
    encode.style.width = "25%";
    encode.style.transition = "width 0.8s";
  }, 800);

  setTimeout(() => {
    verify.style.width = "20%";
    verify.style.transition = "width 0.8s";
  }, 1200);

  setTimeout(() => {
    text.style.opacity = 1;
    text.style.transition = "opacity 0.6s";
  }, 1800);
}

window.addEventListener("load", animateProblem);

setTimeout(() => {
  document.querySelector(".decode").style.width = "0%";
  document.querySelector(".encode").style.width = "0%";
}, 2600);

function animateTakeaways() {
  const items = document.querySelectorAll(".takeaway");

  items.forEach((el, i) => {
    setTimeout(() => {
      el.style.opacity = 1;
      el.style.transform = "translateY(0) scale(1)";
      el.style.transition = "all 0.5s ease";
    }, i * 500); // sequential timing
  });
}

window.addEventListener("load", animateTakeaways);

function flash(id, cls = 'flash') {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), 600);
}

function highlight(id, cls = 'highlighted') {
  const el = document.getElementById(id);
  if (el) el.classList.add(cls);
}
function unhighlight(id, cls = 'highlighted') {
  const el = document.getElementById(id);
  if (el) el.classList.remove(cls);
}
function highlightArrow(id, active = true, saved = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active', 'saved', 'dim');
  if (active) el.classList.add('active');
  else if (saved) el.classList.add('saved');
  else el.classList.add('dim');
}
function resetArrow(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active', 'saved', 'dim');
}

// ── Packet animation ──────────────────────────────────────────────
function animPacket(parentId, fromEl, toEl, color = 'pink', dur = 400) {
  const parent = document.getElementById(parentId);
  const row = parent?.closest('.pipeline-row') || parent?.parentElement;
  if (!fromEl || !toEl || !row) return Promise.resolve();

  return new Promise(resolve => {
    const rowRect = row.getBoundingClientRect();
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();

    const startX = fromRect.right - rowRect.left + 4;
    const endX   = toRect.left  - rowRect.left - 4;
    const y      = (fromRect.top + fromRect.bottom) / 2 - rowRect.top;

    const pkt = document.createElement('div');
    pkt.style.cssText = `
      position:absolute;
      width:10px; height:10px;
      border-radius:50%;
      background:${color === 'teal' ? 'var(--teal)' : 'var(--pink-dark)'};
      top:${y - 5}px;
      left:${startX}px;
      opacity:1;
      z-index:50;
      pointer-events:none;
      transition: left ${dur}ms ease, opacity ${dur * 0.3}ms ease ${dur * 0.7}ms;
    `;
    row.style.position = 'relative';
    row.appendChild(pkt);

    requestAnimationFrame(() => {
      pkt.style.left = endX + 'px';
      setTimeout(() => {
        pkt.style.opacity = '0';
        setTimeout(() => { pkt.remove(); resolve(); }, dur * 0.35);
      }, dur * 0.65);
    });
  });
}

// ── Steps definition ────────────────────────────────────────────────
const stepDescs = [
  'Sampling noise z_T…',
  'DiT generator forward pass…',
  'Decode latent → pixel (Standard) / tap hidden state (VHS)…',
  'CLIP re-encode x₀ (Standard only) / VHS skips this step…',
  'Multimodal LLM scores the candidate…',
  'VHS selects YES, Standard returns NO — 63.3% faster!'
];

// All arrows to reset
const stdArrows = ['arr-s0','arr-s1','arr-s2','arr-s3','arr-s4'];
const vhsArrows = ['arr-v0','arr-v1','arr-v2','arr-v4'];
const allHighlights = [
  'b-s-noise','b-s-gen','b-s-dec','b-s-enc','b-s-llm',
  'b-v-noise','b-v-gen','b-v-hidden','b-v-llm'
];

function resetAll() {
  [...stdArrows, ...vhsArrows].forEach(resetArrow);
  allHighlights.forEach(id => {
    unhighlight(id, 'highlighted');
    unhighlight(id, 'highlighted-teal');
  });
  document.getElementById('badge-s').className = 'outcome-badge badge-no';
  document.getElementById('badge-s').textContent = 'NO ✗';
  document.getElementById('badge-v').className = 'outcome-badge badge-yes';
  document.getElementById('badge-v').textContent = 'YES ✓';
}

// ── Step runner ─────────────────────────────────────────────────────
let currentStep = -1;
let animTimer = null;
let isPlaying = true;
const STEP_DUR = 1600; // ms between steps

function setStep(i) {
  currentStep = i;
  // Update dots
  document.querySelectorAll('.step-dot').forEach(d => {
    d.classList.toggle('active', +d.dataset.i === i);
  });
  document.getElementById('step-desc').textContent = stepDescs[i] || '';
}

async function runStep(step) {
  resetAll();
  setStep(step);

  switch (step) {
    case 0: // Noise
      highlight('b-s-noise');
      highlight('b-v-noise', 'highlighted-teal');
      flash('b-s-noise');
      flash('b-v-noise', 'flash-teal');
      break;

    case 1: // Generator
      highlightArrow('arr-s0', true);
      highlightArrow('arr-v0', false, true);
      await Promise.all([
        animPacket('row-standard',
          document.getElementById('b-s-noise'),
          document.getElementById('b-s-gen'), 'pink', 500),
        animPacket('row-vhs',
          document.getElementById('b-v-noise'),
          document.getElementById('b-v-gen'), 'teal', 500)
      ]);
      highlight('b-s-gen');
      highlight('b-v-gen', 'highlighted-teal');
      flash('b-s-gen');
      flash('b-v-gen', 'flash-teal');
      break;

    case 2: // Decode / hidden state
      highlightArrow('arr-s1', true);
      highlightArrow('arr-v1', false, true);
      await Promise.all([
        animPacket('row-standard',
          document.getElementById('b-s-gen'),
          document.getElementById('b-s-dec'), 'pink', 480),
        animPacket('row-vhs',
          document.getElementById('b-v-gen'),
          document.getElementById('b-v-hidden'), 'teal', 480)
      ]);
      highlight('b-s-dec');
      highlight('b-v-hidden', 'highlighted-teal');
      flash('b-s-dec');
      flash('b-v-hidden', 'flash-teal');
      break;

    case 3: // CLIP encode (Standard) / skip (VHS)
      highlightArrow('arr-s2', true);
      await animPacket('row-standard',
        document.getElementById('b-s-dec'),
        document.getElementById('b-s-enc'), 'pink', 480);
      highlight('b-s-enc');
      flash('b-s-enc');
      // VHS arrows stay teal (already saved)
      highlightArrow('arr-v2', false, true);
      break;

    case 4: // LLM scoring
      highlightArrow('arr-s3', true);
      highlightArrow('arr-v2', false, true);
      await Promise.all([
        animPacket('row-standard',
          document.getElementById('b-s-enc'),
          document.getElementById('b-s-llm'), 'pink', 480),
        animPacket('row-vhs',
          document.getElementById('b-v-hidden'),
          document.getElementById('b-v-llm'), 'teal', 480)
      ]);
      highlight('b-s-llm');
      highlight('b-v-llm', 'highlighted-teal');
      flash('b-s-llm');
      flash('b-v-llm', 'flash-teal');
      break;

    case 5: // Output
      highlightArrow('arr-s4', false, false);
      highlightArrow('arr-v4', false, true);
      document.getElementById('badge-s').className = 'outcome-badge badge-no';
      document.getElementById('badge-s').textContent = '⏱ 277 ms → NO ✗';
      document.getElementById('badge-v').className = 'outcome-badge badge-yes';
      document.getElementById('badge-v').textContent = '⚡ 102 ms → YES ✓';
      break;
  }
}

function nextStep() {
  const next = (currentStep + 1) % 6;
  runStep(next);
}

function startLoop() {
  if (animTimer) clearInterval(animTimer);
  animTimer = setInterval(() => {
    if (isPlaying) nextStep();
  }, STEP_DUR);
}

function toggleAnim() {
  isPlaying = !isPlaying;
  const btn = document.getElementById('btn-play');
  const lbl = document.getElementById('speed-label');
  btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  lbl.textContent = isPlaying ? 'Playing…' : 'Paused';
}

function restartAnim() {
  currentStep = -1;
  resetAll();
  setStep(0);
  runStep(0);
  isPlaying = true;
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('speed-label').textContent = 'Playing…';
}

// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  runStep(0);
  startLoop();
});

