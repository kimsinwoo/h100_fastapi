import type { MedicalChatStructured } from "shared/medical/types";

export const mockMedicalStructured: MedicalChatStructured = {
  summary:
    "증상 설명을 바탕으로 감별 가능성이 높은 항목을 정리했습니다. 반드시 수의사 진료를 받아 정확한 판단을 받으세요.",
  severity: "high",
  explanations: [
    {
      id: "exp-1",
      title: "1위 통증",
      description:
        "근육 긴장이나 통증·부상으로 인한 떨림이 의심됩니다. 통증 부위를 정확히 짚기 어려울 수 있습니다.",
      observationGuide:
        "촉진 시 통증·민감도, 보행 이상, 앉기를 꺼리는지 확인해 보세요.",
    },
    {
      id: "exp-2",
      title: "2위 공포 및 불안",
      description:
        "천둥·낯선 사람 등 자극에 의한 스트레스로 자율신경이 과민 반응하여 심리적 불안과 신체적 떨림이 나타날 수 있습니다.",
      observationGuide:
        "최근 큰 소음·환경 변화 유무, 보호자 옆에서 진정되는지 관찰해 보세요.",
    },
  ],
  emergencyCriteria: {
    items: [
      "호흡 곤란 또는 숨이 가쁘다",
      "의식이 흐리거나 쓰러진다",
      "경련이나 발작이 일어난다",
      "구토나 설사가 심하다",
    ],
  },
  followUpQuestions: [
    { id: "q1", question: "저혈당 예방" },
    { id: "q2", question: "영양 관리" },
  ],
  recommendedActions: {
    primary: "가까운 동물병원 찾기",
    secondary: "응급실 안내",
  },
};
