import { motion } from "framer-motion";
import "../styles/hero.css";
import heroImage from "../assets/hero.png";

function Hero() {
  return (
    <section className="hero">
      <div className="hero-left">
        <motion.h1
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Smart Farming <span>Powered by AI</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          Empowering farmers with AI-powered crop recommendations, weather
          forecasts, disease detection, live mandi prices, and government
          schemes—all in one platform.
        </motion.p>

        <div className="hero-buttons">
          <button className="primary-btn">Get Started</button>
          <button className="secondary-btn">Watch Demo</button>
        </div>
      </div>

      <motion.div
        className="hero-right"
        initial={{ opacity: 0, x: 80 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1 }}
      >
        <img src={heroImage} alt="Smart Farming" />
      </motion.div>
    </section>
  );
}

export default Hero;