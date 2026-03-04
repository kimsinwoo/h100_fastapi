import { Routes, Route } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import GeneratePage from "./pages/GeneratePage";
import VideoPage from "./pages/VideoPage";
import TrainingPage from "./pages/TrainingPage";
import MedicalResultExample from "./pages/MedicalResultExample";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<GeneratePage />} />
      <Route path="/video" element={<VideoPage />} />
      <Route path="/training" element={<TrainingPage />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/medical-example" element={<MedicalResultExample />} />
    </Routes>
  );
}
