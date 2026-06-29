import "../styles/navbar.css";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="logo">
        🌱 AgroSaathi
      </div>

      <ul className="nav-links">
        <li>Home</li>
        <li>Features</li>
        <li>Services</li>
        <li>About</li>
        <li>Contact</li>
      </ul>

      <button className="login-btn">
        Login
      </button>
    </nav>
  );
}

export default Navbar;