// Mappy accounts: sign-in (Firebase Authentication) and saved maps (Firestore).
// Every page that needs accounts imports from this file.
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth, onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword,
  GoogleAuthProvider, signInWithPopup, signOut, sendPasswordResetEmail, updateProfile,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import {
  getFirestore, collection, addDoc, getDocs, deleteDoc, doc, query, orderBy,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAkgrGr-KodjXacHtxGGeXDBXht5J-NW8o",
  authDomain: "mappy-b805b.firebaseapp.com",
  projectId: "mappy-b805b",
  storageBucket: "mappy-b805b.firebasestorage.app",
  messagingSenderId: "374648749097",
  appId: "1:374648749097:web:76ab9a0b7eaf9ae9de7038"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
const db = getFirestore(app);

// ---------- signing in and out ----------

// resolves with the signed-in user, or null, once Firebase has checked
export function currentUser() {
  return new Promise(resolve => {
    const stop = onAuthStateChanged(auth, user => { stop(); resolve(user); });
  });
}

export function watchUser(callback) {
  return onAuthStateChanged(auth, callback);
}

export function signInWithEmail(email, password) {
  return signInWithEmailAndPassword(auth, email, password);
}

export async function createAccount(name, email, password) {
  const result = await createUserWithEmailAndPassword(auth, email, password);
  if (name) await updateProfile(result.user, { displayName: name });
  return result;
}

export function signInWithGoogle() {
  return signInWithPopup(auth, new GoogleAuthProvider());
}

export function resetPassword(email) {
  return sendPasswordResetEmail(auth, email);
}

export function signOutUser() {
  return signOut(auth);
}

// plain-language messages for the errors people are most likely to see
export function friendlyError(err) {
  const messages = {
    "auth/invalid-credential": "That email and password don't match. Please try again.",
    "auth/wrong-password": "That email and password don't match. Please try again.",
    "auth/user-not-found": "That email and password don't match. Please try again.",
    "auth/invalid-email": "That doesn't look like an email address.",
    "auth/email-already-in-use": "There is already an account with that email. Try signing in instead.",
    "auth/weak-password": "Your password needs at least 6 characters.",
    "auth/missing-password": "Please type a password.",
    "auth/too-many-requests": "Too many tries. Please wait a few minutes and try again.",
    "auth/network-request-failed": "Can't connect. Please check your internet connection.",
    "auth/unauthorized-domain": "This website isn't allowed to sign in yet (see Authorized domains in Firebase).",
  };
  return messages[err && err.code] || "Something went wrong. Please try again.";
}

// ---------- saved maps ----------
// Signed in: saved to the account (any device). Not signed in: saved in this browser.
const LIBRARY_KEY = "mappy-library";

function localMaps() {
  try { return JSON.parse(localStorage.getItem(LIBRARY_KEY)) || []; } catch (err) { return []; }
}

export async function saveMap(map) {
  const user = await currentUser();
  if (user) {
    await addDoc(collection(db, "users", user.uid, "maps"), map);
    return "account";
  }
  const list = localMaps();
  list.push({ ...map, id: String(Date.now()) });
  localStorage.setItem(LIBRARY_KEY, JSON.stringify(list));   // throws if the browser storage is full
  return "browser";
}

// newest first
export async function loadMaps() {
  const user = await currentUser();
  if (user) {
    const snap = await getDocs(query(collection(db, "users", user.uid, "maps"), orderBy("date", "desc")));
    return snap.docs.map(d => ({ ...d.data(), id: d.id }));
  }
  return localMaps().reverse();
}

export async function deleteMap(id) {
  const user = await currentUser();
  if (user) {
    await deleteDoc(doc(db, "users", user.uid, "maps", id));
  } else {
    localStorage.setItem(LIBRARY_KEY, JSON.stringify(localMaps().filter(m => String(m.id) !== String(id))));
  }
}
