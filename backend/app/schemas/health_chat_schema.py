"""
건강 상담 AI 구조화 응답 스키마. 1~4순위 감별, 응급 기준, 핵심 질문, 추천 카테고리.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DifferentialItem(BaseModel):
    """감별 진단 1개. rank는 1~4만 허용."""
    rank: int = Field(..., ge=1, le=4, description="1~4순위만 허용")
    name: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1, description="병태생리적 근거")
    emergency: bool = Field(...)
    home_check: str = Field(..., min_length=1, description="보호자 관찰 포인트")


class RecommendedCategory(BaseModel):
    """이어갈 추천 질문 카테고리."""
    label: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class HealthChatStructured(BaseModel):
    """건강 상담 구조화 응답. 5순위 이상·단순 나열 금지."""
    differential: list[DifferentialItem] = Field(..., min_length=1, max_length=4)
    emergency_criteria: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list, max_length=5)
    recommended_categories: list[RecommendedCategory] = Field(..., min_length=2)

