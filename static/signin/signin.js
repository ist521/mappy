import {
  signInWithEmail, createAccount, signInWithGoogle, resetPassword, friendlyError,
} from "../firebase/firebase.js";

const params = new URLSearchParams(location.search);
const creating = params.get("mode") === "create";

const form = document.getElementById("signin-form");
const nameInput = document.getElementById("name");
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const message = document.getElementById("message");
const submitBtn = document.getElementById("submit-btn");

function showMessage(text, isError = true) {
  message.textContent = text;
  message.className = isError ? "error" : "info";
  message.hidden = false;
}

function goHome() {
  location.href = "homepage.html";
}

// "create an account" uses the same page with a name field and different wording
if (creating) {
  document.title = "Create an account · Mappy";
  document.getElementById("signin-title").textContent = "Create an account";
  document.getElementById("name-row").hidden = false;
  document.getElementById("google-btn").textContent = "Sign up with Google";
  submitBtn.textContent = "Create account";
  passwordInput.autocomplete = "new-password";
  document.getElementById("forgot-btn").hidden = true;
  document.getElementById("switch-text").textContent = "Already have an account?";
  const link = document.getElementById("switch-link");
  link.textContent = "Sign in";
  link.href = "signin.html";
}

document.getElementById("show-password").onchange = (e) => {
  passwordInput.type = e.target.checked ? "text" : "password";
};

form.onsubmit = async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  message.hidden = true;
  try {
    if (creating) {
      await createAccount(nameInput.value.trim(), emailInput.value.trim(), passwordInput.value);
    } else {
      await signInWithEmail(emailInput.value.trim(), passwordInput.value);
    }
    goHome();
  } catch (err) {
    console.error(err);
    showMessage(friendlyError(err));
    submitBtn.disabled = false;
  }
};

document.getElementById("google-btn").onclick = async () => {
  message.hidden = true;
  try {
    await signInWithGoogle();
    goHome();
  } catch (err) {
    if (err.code === "auth/popup-closed-by-user" || err.code === "auth/cancelled-popup-request") return;
    console.error(err);
    showMessage(friendlyError(err));
  }
};

document.getElementById("forgot-btn").onclick = async () => {
  const email = emailInput.value.trim();
  if (!email) {
    showMessage("Type your email above first, then press \"Forgot password?\" again.");
    emailInput.focus();
    return;
  }
  try {
    await resetPassword(email);
    showMessage("If there's an account with that email, we sent a link to reset the password.", false);
  } catch (err) {
    showMessage(friendlyError(err));
  }
};
