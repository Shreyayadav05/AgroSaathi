function Hero() {
  return (
    <section
      style={{
        textAlign: "center",
        padding: "100px 20px",
        background: "#e8f5e9",
      }}
    >
      <h1 style={{ fontSize: "48px", marginBottom: "20px" }}>
        🌾 Smart Farming Powered by AI
      </h1>

      <p
        style={{
          fontSize: "20px",
          color: "#555",
          maxWidth: "700px",
          margin: "auto",
        }}
      >
        Empowering farmers with AI-powered crop recommendations, weather
        insights, disease detection, live market prices, and government schemes.
      </p>

      <button
        style={{
          marginTop: "40px",
          padding: "15px 40px",
          border: "none",
          borderRadius: "10px",
          background: "#2e7d32",
          color: "white",
          fontSize: "18px",
          cursor: "pointer",
        }}
      >
        Get Started
      </button>
    </section>
  );
}

export default Hero;