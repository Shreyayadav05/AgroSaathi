import "../styles/stats.css";
import { FaUsers, FaSeedling, FaRobot, FaMapMarkedAlt } from "react-icons/fa";

function Stats() {
  const stats = [
    {
      icon: <FaUsers />,
      number: "25K+",
      title: "Farmers Helped",
    },
    {
      icon: <FaSeedling />,
      number: "120+",
      title: "Supported Crops",
    },
    {
      icon: <FaRobot />,
      number: "1M+",
      title: "AI Predictions",
    },
    {
      icon: <FaMapMarkedAlt />,
      number: "500+",
      title: "Villages Connected",
    },
  ];

  return (
    <section className="stats-section">
      <h2>AgroSaathi in Numbers</h2>

      <div className="stats-grid">
        {stats.map((item, index) => (
          <div className="stat-card" key={index}>
            <div className="stat-icon">
              {item.icon}
            </div>

            <h1>{item.number}</h1>

            <p>{item.title}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default Stats;