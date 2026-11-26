// Dataset tabs: only the visible panel plays, and its front and chase views start together.
for (const tabs of document.querySelectorAll('.tabs')) {
  const panels = tabs.parentElement.querySelectorAll('.tab-panel');
  const show = (name) => {
    for (const tab of tabs.querySelectorAll('.tab')) {
      const on = tab.dataset.tab === name;
      tab.classList.toggle('active', on);
      tab.setAttribute('aria-selected', String(on));
    }
    for (const panel of panels) {
      const on = panel.dataset.panel === name;
      panel.hidden = !on;
      for (const video of panel.querySelectorAll('video')) {
        if (on) {
          video.currentTime = 0;
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      }
    }
  };
  tabs.addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (tab) show(tab.dataset.tab);
  });
  show(tabs.querySelector('.tab.active').dataset.tab);
}
