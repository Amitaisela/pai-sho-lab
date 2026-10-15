/* Move sounds and haptics.
 *
 * The sounds are synthesised with WebAudio, so there are no audio files to host or license.
 * Sound is on by default where there is a mouse and off on touch screens (a phone is often in a
 * quiet place; there is no way to read the silent switch from a web page); haptics (where the
 * browser has navigator.vibrate) are on. Both are per-browser settings in localStorage, toggled
 * from the menu, and every storage access is guarded (private windows throw).
 */
const SOUND_KEY = 'pai_sound';
const HAPTICS_KEY = 'pai_haptics';

function read(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function write(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode */ }
}

function coarsePointer() {
  return !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
}

export function soundOn() {
  const v = read(SOUND_KEY);
  return v === null ? !coarsePointer() : v === 'on';
}
export function setSound(on) { write(SOUND_KEY, on ? 'on' : 'off'); }

export function canVibrate() { return typeof navigator.vibrate === 'function'; }
export function hapticsOn() { return read(HAPTICS_KEY) !== 'off'; }
export function setHaptics(on) { write(HAPTICS_KEY, on ? 'on' : 'off'); }

/** A short vibration (ms, or a pattern) where the device supports it and haptics are on. */
export function buzz(pattern) {
  if (hapticsOn() && canVibrate()) navigator.vibrate(pattern);
}

let audio = null;
function context() {
  if (!audio) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audio = new Ctx();
  }
  if (audio.state === 'suspended') audio.resume();
  return audio;
}

/** A pitched blip that falls from `freq` to `to`: the body of a tile set down on a board. */
function blip(c, t, { freq, to = freq, dur, gain, type = 'sine' }) {
  const osc = c.createOscillator();
  const amp = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t);
  osc.frequency.exponentialRampToValueAtTime(Math.max(30, to), t + dur);
  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.exponentialRampToValueAtTime(gain, t + 0.004);
  amp.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(amp).connect(c.destination);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

/** A burst of filtered noise: the click of the contact. */
function tick(c, t, { gain, dur, freq = 2200 }) {
  const frames = Math.max(1, Math.floor(c.sampleRate * dur));
  const buffer = c.createBuffer(1, frames, c.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / frames);
  const src = c.createBufferSource();
  const filter = c.createBiquadFilter();
  const amp = c.createGain();
  filter.type = 'bandpass';
  filter.frequency.value = freq;
  amp.gain.value = gain;
  src.buffer = buffer;
  src.connect(filter).connect(amp).connect(c.destination);
  src.start(t);
}

const SOUNDS = {
  move: (c, t) => { blip(c, t, { freq: 210, to: 120, dur: 0.09, gain: 0.3 }); tick(c, t, { gain: 0.25, dur: 0.03 }); },
  capture: (c, t) => {
    blip(c, t, { freq: 150, to: 70, dur: 0.16, gain: 0.45 });
    tick(c, t, { gain: 0.5, dur: 0.07, freq: 1400 });
  },
  lowtime: (c, t) => { blip(c, t, { freq: 880, dur: 0.09, gain: 0.2 }); },
  end: (c, t) => {
    blip(c, t, { freq: 523, dur: 0.22, gain: 0.22 });
    blip(c, t + 0.12, { freq: 392, dur: 0.34, gain: 0.22 });
  },
};

/** Plays `name` ('move', 'capture', 'lowtime' or 'end') when sound is on. Never throws. */
export function play(name) {
  if (!soundOn() || !SOUNDS[name]) return;
  try {
    const c = context();
    if (c) SOUNDS[name](c, c.currentTime);
  } catch { /* an audio device that is missing or refused must not break the game */ }
}
