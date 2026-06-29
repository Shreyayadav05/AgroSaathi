function Navbar() {
  return (
    <nav
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "20px 60px",
        background: "#1b5e20",
        color: "white",
      }}
    >
      <h2>🌱 AgroSaathi</h2>

      <div style={{ display: "flex", gap: "30px" }}>
        <a href="#" style={{ color: "white", textDecoration: "none" }}>Home</a>
        <a href="#" style={{ color: "white", textDecoration: "none" }}>Features</a>
        <a href="#" style={{ color: "white", textDecoration: "none" }}>About</a>
        <a href="#" style={{ color: "white", textDecoration: "none" }}>Contact</a>
      </div>
    </nav>
  );
}

export default Navbar;