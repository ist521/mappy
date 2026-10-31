// shows "Signed in as …  Sign out" (or a sign-in link) on the homepage
import { watchUser, signOutUser } from "../firebase/firebase.js";

const bar = document.getElementById("account-bar");

watchUser(user => {
  bar.innerHTML = "";
  if (user) {
    const who = document.createElement("span");
    who.textContent = `Signed in as ${user.displayName || user.email}`;
    const out = document.createElement("button");
    out.textContent = "sign out";
    out.onclick = async () => { await signOutUser(); location.href = "index.html"; };
    bar.append(who, " ", out);
  } else {
    const link = document.createElement("a");
    link.href = "signin.html";
    link.textContent = "Sign in to keep your maps on every device";
    bar.append(link);
  }
});
