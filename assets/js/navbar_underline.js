const navbar = document.querySelector(".navbar-nav");
const navLinks = navbar.querySelectorAll(".nav-link");
const activeNavItem = navbar.querySelector(".nav-item.active > .nav-link");
const navbarToggler = document.querySelector(".navbar-toggler");

let flag = true;

let start_pos = navLinks[0].offsetLeft;
let active_nav_width = activeNavItem.offsetWidth;
let active_nav_offset = activeNavItem.offsetLeft;

function setUnderlineStyles(width, offset) {
  if (flag) {
    navbar.style.setProperty("--underline-width", `${0.9 * width}px`);
    navbar.style.setProperty("--underline-offset-x", `${offset - start_pos + 0.05 * width}px`);
  } else {
    navbar.style.setProperty("--underline-width", `0px`);
    navbar.style.setProperty("--underline-offset-x", `0px`);
  }
}

function updateUnderlineStyles() {
  setUnderlineStyles(active_nav_width, active_nav_offset);
}

function handleNavbarMouseover(event) {
  if (flag && event.target.classList.contains("nav-link")) {
    setUnderlineStyles(event.target.offsetWidth, event.target.offsetLeft);
  }
}

function handleNavbarMouseleave() {
  updateUnderlineStyles();
}

function handleWindowResize() {
  start_pos = navLinks[0].offsetLeft;
  active_nav_width = activeNavItem.offsetWidth;
  active_nav_offset = activeNavItem.offsetLeft;

  if (window.innerWidth < 576) {
    flag = false;
  } else {
    flag = true;
  }

  updateUnderlineStyles();
}

function handleClickToggler() {
  flag = false;
}

navbarToggler.addEventListener("click", handleClickToggler);

// Initial setup
updateUnderlineStyles();

// Event listeners
navbar.addEventListener("mouseover", handleNavbarMouseover);
navbar.addEventListener("mouseleave", handleNavbarMouseleave);
window.addEventListener("resize", handleWindowResize, 100);
