// main.js - Client-side functionality for Agriculture Analytics Platform

document.addEventListener("DOMContentLoaded", function () {
  // Enable tooltips
  const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltips.forEach((tooltip) => {
    new bootstrap.Tooltip(tooltip);
  });

  // Enable popovers
  const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
  popovers.forEach((popover) => {
    new bootstrap.Popover(popover);
  });

  // For the prediction form
  const predictionForm = document.getElementById("prediction-form");
  if (predictionForm) {
    predictionForm.addEventListener("submit", function () {
      const submitBtn = this.querySelector('button[type="submit"]');
      submitBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
      submitBtn.disabled = true;
    });
  }

  // For any flash messages or alerts
  const alerts = document.querySelectorAll(".alert-dismissible");
  alerts.forEach((alert) => {
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      const closeBtn = alert.querySelector(".btn-close");
      if (closeBtn) {
        closeBtn.click();
      } else {
        alert.classList.add("fade");
        setTimeout(() => alert.remove(), 500);
      }
    }, 5000);
  });

  // For chart resizing
  function resizeCharts() {
    if (typeof Plotly !== "undefined") {
      const charts = document.querySelectorAll('[id^="plotly-"]');
      charts.forEach((chart) => {
        Plotly.relayout(chart, {
          width: chart.offsetWidth,
          height: chart.offsetHeight,
        });
      });
    }
  }

  // Add resize event listener
  window.addEventListener("resize", function () {
    // Debounce the resize event
    clearTimeout(window.resizeTimer);
    window.resizeTimer = setTimeout(resizeCharts, 250);
  });

  // Activate current navigation link
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll(".navbar-nav .nav-link");

  navLinks.forEach((link) => {
    const linkPath = link.getAttribute("href");
    if (currentPath === linkPath) {
      link.classList.add("active");
    }
  });

  // For scrolling animations
  const scrollElements = document.querySelectorAll(".scroll-fade");

  const elementInView = (el, divider = 1) => {
    const elementTop = el.getBoundingClientRect().top;
    return (
      elementTop <=
      (window.innerHeight || document.documentElement.clientHeight) / divider
    );
  };

  const displayScrollElement = (element) => {
    element.classList.add("fade-in");
  };

  const hideScrollElement = (element) => {
    element.classList.remove("fade-in");
  };

  const handleScrollAnimation = () => {
    scrollElements.forEach((el) => {
      if (elementInView(el, 1.2)) {
        displayScrollElement(el);
      } else {
        hideScrollElement(el);
      }
    });
  };

  window.addEventListener("scroll", () => {
    handleScrollAnimation();
  });

  // Initial check for elements in view
  handleScrollAnimation();
});
