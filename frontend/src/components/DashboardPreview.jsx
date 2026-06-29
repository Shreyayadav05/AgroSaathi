import "../styles/dashboard.css";
import {
  FaRobot,
  FaCloudSun,
  FaLeaf,
  FaMoneyBillWave
} from "react-icons/fa";

function DashboardPreview() {
  return (
    <section className="dashboard">

      <h2>Farmer Dashboard</h2>

      <div className="dashboard-grid">

        <div className="dashboard-card">
          <FaRobot className="dashboard-icon"/>
          <h3>AI Assistant</h3>
          <p>Ask anything about crops, fertilizers and farming.</p>
        </div>

        <div className="dashboard-card">
          <FaCloudSun className="dashboard-icon"/>
          <h3>Weather</h3>
          <h1>28°C</h1>
          <p>Sunny</p>
        </div>

        <div className="dashboard-card">
          <FaMoneyBillWave className="dashboard-icon"/>
          <h3>Today's Mandi</h3>
          <h1>₹2450</h1>
          <p>Paddy / Quintal</p>
        </div>

        <div className="dashboard-card">
          <FaLeaf className="dashboard-icon"/>
          <h3>Crop Health</h3>
          <h1>Healthy</h1>
          <p>98% Confidence</p>
        </div>

      </div>

    </section>
  );
}

export default DashboardPreview;