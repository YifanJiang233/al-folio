window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    macros: {
      md: ["{\\mathop{}\\mathopen{\\mathrm{d}}}"],
      bR: ["{\\mathbf{R}}"],
      bbR: ["{\\mathbb{R}}"],
      E: ["{\\mathbf{E}}"],
      bbP: ["{\\mathbb{P}}"],
    },
  },
  svg: {
    fontCache: "global",
  },
  // output: {font: 'mathjax-modern'}
};
