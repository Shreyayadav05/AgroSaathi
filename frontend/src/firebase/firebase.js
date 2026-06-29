import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDwx42UfFRD9etB7AGY6XBJ1bB4EGIZoAo",
  authDomain: "agrosaathi-f39cd.firebaseapp.com",
  projectId: "agrosaathi-f39cd",
  storageBucket: "agrosaathi-f39cd.firebasestorage.app",
  messagingSenderId: "251611518733",
  appId: "1:251611518733:web:d484aeeb47daf4cd284244",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);