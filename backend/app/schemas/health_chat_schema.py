"""
건강 상담 AI 구조화 응답 스키마. 4순위 감별 필수, 병태생리·관찰 포인트 2~3문장 이상.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# 감별 1개당 reason·home_check 최소 길이 (2~3문장 수준, 설명 누락 시 검증 오류)
MIN_REASON_LENGTH = 40
MIN_HOME_CHECK_LENGTH = 25


class DifferentialItem(BaseModel):
    """감별 진단 1개. rank 1~4, reason·home_check는 최소 2~3문장 수준."""
    rank: int = Field(..., ge=1, le=4, description="1~4순위")
    name: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=MIN_REASON_LENGTH, description="병태생리적 근거, 최소 2~3문장")
    emergency: bool = Field(...)
    home_check: str = Field(..., min_length=MIN_HOME_CHECK_LENGTH, description="보호자 관찰 포인트, 이해 가능하게")


class RecommendedCategory(BaseModel):
    """이어갈 추천 질문 카테고리."""
    label: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class HealthChatStructured(BaseModel):
    """건강 상담 구조화 응답. 4순위까지 반드시 출력, 설명 누락 시 오류."""
    differential: list[DifferentialItem] = Field(..., min_length=4, max_length=4, description="반드시 4개")
    emergency_criteria: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list, max_length=5)
    recommended_categories: list[RecommendedCategory] = Field(..., min_length=2)

