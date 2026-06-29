import "../styles/features.css";
import {
  FaRobot,
  FaCloudSun,
  FaLeaf,
  FaCamera,
  FaMoneyBillWave,
  FaUniversity,
} from "react-icons/fa";

const features = [
  {
    icon: <FaRobot />,
    title: "AI Farming Assistant",
    desc: "Ask farming questions in your language and receive intelligent guidance.",
  },
  {
    icon: <FaCloudSun />,
    title: "Weather Forecast",
    desc: "Get real-time weather forecasts to plan irrigation and harvesting.",
  },
  {
    icon: <FaLeaf />,
    title: "Crop Recommendation",
    desc: "Receive crop suggestions based on soil and seasonal conditions.",
  },
  {
    icon: <FaCamera />,
    title: "Disease Detection",
    desc: "Upload crop images and detect diseases using AI.",
  },
  {
    icon: <FaMoneyBillWave />,
    title: "Live Mandi Prices",
    desc: "Track the latest market prices to sell crops at the right time.",
  },
  {
    icon: <FaUniversity />,
    title: "Government Schemes",
    desc: "Discover subsidies, loans, and schemes available for farmers.",
  },
];

function Features() {
  return (
    <section className="features">
      <h2>Our Smart Features</h2>

      <div className="feature-grid">
        {features.map((item, index) => (
          <div className="feature-card" key={index}>
            <div className="icon">{item.icon}</div>
            <h3>{item.title}</h3>
            <p>{item.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default Features;