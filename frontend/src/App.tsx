import { Routes, Route } from "react-router-dom";
import GeneratePage from "./pages/GeneratePage";
import TrainingPage from "./pages/TrainingPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<GeneratePage />} />
      <Route path="/training" element={<TrainingPage />} />
    </Routes>
  );
}
