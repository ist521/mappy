// Speak-to-type: press the microphone, say a word, then pick what you said.
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById("mic-btn");
const speechBox = document.getElementById("speech-box");
const speechStatus = document.getElementById("speech-status");
const speechChoices = document.getElementById("speech-choices");
const wordInput = document.getElementById("textentry");

let recognition = null;
let listening = false;

function goToMap(word) {
  wordInput.value = word;
  location.href = "map.html?word=" + encodeURIComponent(word);
}

function showStatus(text) {
  speechBox.hidden = false;
  speechStatus.textContent = text;
}

function showChoices(words) {
  speechChoices.innerHTML = "";
  words.forEach(w => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "speech-choice";
    btn.textContent = w;
    btn.onclick = () => goToMap(w);
    speechChoices.appendChild(btn);
  });
  const again = document.createElement("button");
  again.type = "button";
  again.className = "speech-again";
  again.textContent = "🎤 try again";
  again.onclick = startListening;
  speechChoices.appendChild(again);
}

function startListening() {
  if (listening) { recognition.stop(); return; }
  speechChoices.innerHTML = "";
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 5;      // several guesses, so the person can pick the right one

  recognition.onstart = () => {
    listening = true;
    micBtn.classList.add("listening");
    micBtn.textContent = "■ stop";
    showStatus("Listening… say a word.");
  };

  recognition.onresult = (event) => {
    const guesses = [];
    for (const alt of event.results[0]) {
      const w = alt.transcript.trim().toLowerCase().replace(/[.,!?]/g, "");
      if (w && !guesses.includes(w)) guesses.push(w);
    }
    if (!guesses.length) { showStatus("I didn't catch that. Try again."); showChoices([]); return; }
    wordInput.value = guesses[0];       // also fills the box, so it can be fixed by typing
    showStatus("Did you say:");
    showChoices(guesses);
  };

  recognition.onerror = (event) => {
    const messages = {
      "not-allowed": "Please allow the microphone (click the 🔒 or 🎤 icon in the address bar), then try again.",
      "service-not-allowed": "Please allow the microphone, then try again.",
      "no-speech": "I didn't hear anything. Try again.",
      "audio-capture": "No microphone was found.",
      "network": "Speech needs an internet connection. Please check it and try again.",
      "aborted": "",
    };
    const msg = event.error in messages ? messages[event.error] : "Something went wrong. Please try again.";
    if (msg) { showStatus(msg); showChoices([]); }
  };

  recognition.onend = () => {
    listening = false;
    micBtn.classList.remove("listening");
    micBtn.textContent = "🎤 say a word";
  };

  recognition.start();
}

if (SpeechRecognition) {
  micBtn.onclick = startListening;
} else {
  // e.g. Firefox: no built-in speech recognition, so hide the button
  micBtn.hidden = true;
  document.getElementById("mic-note").hidden = true;
}
