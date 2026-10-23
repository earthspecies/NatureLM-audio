function enhanceContent() {
  const arrowSVG = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';
  document.querySelectorAll('.arrow-link, .ext-btn').forEach(function (el) {
    if (el.querySelector('svg')) return;
    el.insertAdjacentHTML('beforeend', arrowSVG);
  });

  const copyIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  const checkIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

  document.querySelectorAll('.prompt-block').forEach(function (block) {
    if (block.querySelector('.prompt-copy-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'prompt-copy-btn';
    btn.innerHTML = copyIcon;

    btn.addEventListener('click', function () {
      const text = block.dataset.prompt !== undefined ? block.dataset.prompt : block.textContent.trim();
      navigator.clipboard.writeText(text).then(function () {
        btn.innerHTML = checkIcon;
        btn.style.opacity = '1';
        setTimeout(function () {
          btn.innerHTML = copyIcon;
          btn.style.opacity = '';
        }, 1500);
      });
    });

    block.appendChild(btn);
  });
}

// This file loads outside Material's content container, so instant navigation
// never re-executes it and DOMContentLoaded fires only once. Material's
// document$ observable emits on first load and after every instant page swap.
if (typeof document$ !== 'undefined') {
  document$.subscribe(enhanceContent);
} else {
  document.addEventListener('DOMContentLoaded', enhanceContent);
}
