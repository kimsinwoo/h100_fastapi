import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  getLlmStatus,
  llmChat,
  getErrorMessage,
  getChatRooms,
  getChatRoom,
  createChatRoom,
  addChatMessage,
  deleteChatRoom,
} from "../services/api";
import type { ChatRoomSummary, HealthChatStructured } from "../services/api";

type Message = { role: "user" | "assistant"; content: string; structured?: HealthChatStructured };

/** 건강 도우미 응답: **굵게** 처리 + 줄바꿈·• 목록 유지 */
function renderAssistantContent(content: string) {
  const parts = content.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span className="whitespace-pre-wrap">
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          part
        )
      )}
    </span>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [available, setAvailable] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [rooms, setRooms] = useState<ChatRoomSummary[]>([]);
  const [currentRoomId, setCurrentRoomId] = useState<string | null>(null);
  const [roomsLoading, setRoomsLoading] = useState(true);
  const [waitingSeconds, setWaitingSeconds] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<Message[]>([]);
  const sendingRef = useRef(false);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // LLM 응답 대기 중 경과 시간 표시 (최대 5분 안내)
  useEffect(() => {
    if (!loading) {
      setWaitingSeconds(0);
      return;
    }
    const start = Date.now();
    const t = setInterval(() => setWaitingSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(t);
  }, [loading]);

  const loadRooms = useCallback(async () => {
    setRoomsLoading(true);
    try {
      const list = await getChatRooms();
      setRooms(list);
    } catch {
      setRooms([]);
    } finally {
      setRoomsLoading(false);
    }
  }, []);

  const loadRoom = useCallback(async (roomId: string) => {
    try {
      const room = await getChatRoom(roomId);
      setMessages((room.messages || []).map((m) => ({ role: m.role as "user" | "assistant", content: m.content })));
      setCurrentRoomId(roomId);
    } catch {
      setMessages([]);
      setCurrentRoomId(roomId);
    }
  }, []);

  useEffect(() => {
    getLlmStatus()
      .then((r) => {
        setAvailable(r.available);
        setModel(r.model ?? null);
      })
      .catch(() => setAvailable(false));
  }, []);

  useEffect(() => {
    loadRooms();
  }, [loadRooms]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const startNewChat = useCallback(() => {
    setCurrentRoomId(null);
    setMessages([]);
  }, []);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || !available || loading || sendingRef.current) return;
    sendingRef.current = true;
    setInput("");
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    let activeRoomId: string | null = currentRoomId;
    try {
      if (!activeRoomId) {
        const room = await createChatRoom();
        activeRoomId = room.id;
        setCurrentRoomId(activeRoomId);
        await loadRooms();
      }
      await addChatMessage(activeRoomId!, "user", text);
      // 이어하기 시 항상 최신 메시지 목록 사용 (ref로 클로저 문제 방지)
      const latest = [...messagesRef.current, userMsg];
      const history = latest.map((m) => ({ role: m.role, content: m.content }));
      let response: Awaited<ReturnType<typeof llmChat>>;
      try {
        response = await llmChat(history, 1024, 0.4);
      } catch (firstErr) {
        const msg = String((firstErr as Error)?.message ?? "").toLowerCase();
        const isRetryable = msg.includes("network") || msg.includes("timeout") || msg.includes("초과") || msg.includes("econnaborted");
        if (isRetryable) {
          response = await llmChat(history, 1024, 0.4);
        } else {
          throw firstErr;
        }
      }
      let assistantContent = response?.content != null && String(response.content).trim() !== "" ? String(response.content).trim() : "응답을 생성하지 못했습니다. 다시 시도해 주세요.";
      if (response?.structured) {
        assistantContent = assistantContent.replace(/\s*```(?:json)?\s*[\s\S]*?\s*```\s*$/, "").trim() || assistantContent;
      }
      const assistantMsg: Message = { role: "assistant", content: assistantContent, structured: response?.structured };
      setMessages((prev) => [...prev, assistantMsg]);
      if (activeRoomId) {
        try {
          await addChatMessage(activeRoomId, "assistant", assistantContent);
        } catch (e) {
          try {
            await addChatMessage(activeRoomId, "assistant", assistantContent);
          } catch {
            // 저장 실패해도 화면에는 이미 표시됨
          }
        }
      }
      await loadRooms();
    } catch (err) {
      const errMsg = `오류: ${getErrorMessage(err)}`;
      setMessages((prev) => [...prev, { role: "assistant", content: errMsg }]);
      if (activeRoomId) {
        try {
          await addChatMessage(activeRoomId, "assistant", errMsg);
        } catch {
          // ignore
        }
      }
    } finally {
      setLoading(false);
      sendingRef.current = false;
    }
  }, [input, available, loading, currentRoomId, loadRooms]);

  /** 추천 질문 버튼 클릭: 해당 query를 사용자 메시지로 추가 후 API 재호출(컨텍스트 유지). 중복 호출 방지. */
  const sendWithQuery = useCallback(
    async (query: string) => {
      const q = query?.trim();
      if (!q || !available || loading || sendingRef.current) return;
      sendingRef.current = true;
      const userMsg: Message = { role: "user", content: q };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      let activeRoomId: string | null = currentRoomId;
      try {
        if (!activeRoomId) {
          const room = await createChatRoom();
          activeRoomId = room.id;
          setCurrentRoomId(activeRoomId);
          await loadRooms();
        }
        await addChatMessage(activeRoomId!, "user", q);
        const latest = [...messagesRef.current, userMsg];
        const history = latest.map((m) => ({ role: m.role, content: m.content }));
        const response = await llmChat(history, 1024, 0.4);
        let assistantContent = response?.content != null && String(response.content).trim() !== "" ? String(response.content).trim() : "응답을 생성하지 못했습니다. 다시 시도해 주세요.";
        if (response?.structured) {
          assistantContent = assistantContent.replace(/\s*```(?:json)?\s*[\s\S]*?\s*```\s*$/, "").trim() || assistantContent;
        }
        const assistantMsg: Message = { role: "assistant", content: assistantContent, structured: response?.structured };
        setMessages((prev) => [...prev, assistantMsg]);
        if (activeRoomId) {
          try {
            await addChatMessage(activeRoomId, "assistant", assistantContent);
          } catch {
            // ignore
          }
        }
        await loadRooms();
      } catch (err) {
        const errMsg = `오류: ${getErrorMessage(err)}`;
        setMessages((prev) => [...prev, { role: "assistant", content: errMsg }]);
        if (activeRoomId) {
          try {
            await addChatMessage(activeRoomId, "assistant", errMsg);
          } catch {
            // ignore
          }
        }
      } finally {
        setLoading(false);
        sendingRef.current = false;
      }
    },
    [available, loading, currentRoomId, loadRooms]
  );

  const handleDeleteRoom = useCallback(
    async (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      if (!confirm("이 채팅방을 삭제할까요?")) return;
      try {
        await deleteChatRoom(id);
        if (currentRoomId === id) {
          startNewChat();
        }
        await loadRooms();
      } catch {
        // ignore
      }
    },
    [currentRoomId, startNewChat, loadRooms]
  );

  const [roomDrawerOpen, setRoomDrawerOpen] = useState(false);

  const closeDrawerAndLoadRoom = useCallback((roomId: string) => {
    loadRoom(roomId);
    setRoomDrawerOpen(false);
  }, [loadRoom]);

  const closeDrawerAndNewChat = useCallback(() => {
    startNewChat();
    setRoomDrawerOpen(false);
  }, [startNewChat]);

  return (
    <div className="min-h-screen bg-gray-100 py-3 md:py-6 pb-[env(safe-area-inset-bottom)] md:pb-6">
      <div className="mx-auto flex max-w-5xl gap-4 px-3 md:px-4">
        {/* 모바일: 채팅방 목록 드로어 */}
        <>
          <div
            role="presentation"
            className={`fixed inset-0 z-20 bg-black/40 md:hidden ${roomDrawerOpen ? "block" : "hidden"}`}
            onClick={() => setRoomDrawerOpen(false)}
            aria-hidden="true"
          />
          <aside
            className={`fixed left-0 top-0 z-30 h-full w-[min(100vw-3rem,18rem)] shrink-0 rounded-r-xl bg-white shadow-xl transition-transform duration-200 ease-out md:static md:z-auto md:h-auto md:w-52 md:rounded-xl md:shadow md:transition-none ${
              roomDrawerOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
            }`}
          >
            <div className="flex h-full flex-col border-b border-gray-200 md:border-b">
              <div className="flex items-center justify-between border-b border-gray-200 p-3">
                <span className="text-sm font-medium text-gray-700 md:sr-only">채팅방</span>
                <button
                  type="button"
                  onClick={closeDrawerAndNewChat}
                  className="flex-1 rounded-lg bg-indigo-600 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 active:bg-indigo-800 md:py-2"
                >
                  새 채팅
                </button>
                <button
                  type="button"
                  onClick={() => setRoomDrawerOpen(false)}
                  className="ml-2 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 md:hidden"
                  aria-label="닫기"
                >
                  ×
                </button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-2 md:max-h-[calc(100vh-12rem)]">
                {roomsLoading ? (
                  <p className="p-2 text-center text-xs text-gray-500">불러오는 중…</p>
                ) : rooms.length === 0 ? (
                  <p className="p-2 text-center text-xs text-gray-500">저장된 채팅이 없습니다.</p>
                ) : (
                  rooms.map((r) => (
                    <div
                      key={r.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => closeDrawerAndLoadRoom(r.id)}
                      onKeyDown={(e) => e.key === "Enter" && closeDrawerAndLoadRoom(r.id)}
                      className={`group flex items-center justify-between gap-1 rounded-lg px-2 py-3 text-left text-sm hover:bg-gray-50 active:bg-gray-100 ${
                        currentRoomId === r.id ? "bg-indigo-50 text-indigo-700" : "text-gray-700"
                      }`}
                    >
                      <span className="min-w-0 flex-1 truncate" title={r.title}>
                        {r.title}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => handleDeleteRoom(e, r.id)}
                        className="shrink-0 rounded p-2 text-gray-400 hover:bg-gray-200 hover:text-red-600 group-hover:opacity-100 touch-manipulation"
                        aria-label="삭제"
                      >
                        ×
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>
        </>

        <div className="min-w-0 flex-1 flex flex-col">
          <header className="mb-3 md:mb-4 flex items-center gap-2 md:gap-4">
            <button
              type="button"
              onClick={() => setRoomDrawerOpen(true)}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 active:bg-gray-100 md:hidden touch-manipulation"
              aria-label="채팅방 목록"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="min-w-0 flex-1">
              <h1 className="text-lg font-bold text-gray-900 md:text-2xl">건강 질문 도우미</h1>
              <p className="mt-0.5 line-clamp-2 text-xs text-gray-500 md:mt-1 md:line-clamp-none md:text-sm">
                {available
                  ? "증상·반려동물 상황을 적어 주세요. 답변에 2~5분 걸릴 수 있어요. 참고 정보만 제공합니다."
                  : "LLM을 사용할 수 없습니다."}
              </p>
            </div>
            <Link
              to="/"
              className="shrink-0 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 active:bg-gray-100 md:px-4 touch-manipulation"
            >
              이미지
            </Link>
          </header>

          <div className="flex min-h-0 flex-1 flex-col rounded-xl bg-white shadow md:min-h-0">
            <div className="flex min-h-[50vh] flex-1 flex-col overflow-y-auto overscroll-contain p-3 md:h-[60vh] md:p-4">
              {messages.length === 0 && !loading && (
                <p className="py-6 text-center text-sm text-gray-500 md:py-4">
                  증상이나 반려동물 상황을 간단히 적어 주세요. (참고 정보만 제공합니다)
                </p>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`mb-3 flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                  <div
                    className={`max-w-[90%] rounded-2xl px-3 py-2.5 text-sm md:max-w-[85%] md:rounded-lg md:px-3 md:py-2 ${
                      m.role === "user"
                        ? "bg-indigo-600 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {m.role === "assistant" ? renderAssistantContent(m.content) : m.content}
                  </div>
                  {m.role === "assistant" && m.structured && (
                    <div className="mt-2 w-full max-w-[90%] space-y-2 md:max-w-[85%]">
                      {m.structured.differential?.length > 0 && (
                        <div className="rounded-lg border border-gray-200 bg-white p-2 text-xs md:text-sm">
                          <p className="mb-1 font-semibold text-gray-700">감별 진단 (1~4순위)</p>
                          <ul className="list-inside list-disc space-y-1">
                            {m.structured.differential.map((d) => (
                              <li key={d.rank}>
                                <strong>{d.rank}위 {d.name}</strong>
                                {d.emergency && <span className="ml-1 text-red-600">(응급 가능)</span>}
                                <br />
                                <span className="text-gray-600">{d.reason}</span>
                                <br />
                                <span className="text-gray-500">관찰: {d.home_check}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {m.structured.emergency_criteria?.length > 0 && (
                        <div className="rounded-lg border border-red-100 bg-red-50/50 p-2 text-xs md:text-sm">
                          <p className="mb-1 font-semibold text-red-800">즉시 병원 내원 기준</p>
                          <ul className="list-inside list-disc space-y-0.5 text-gray-700">
                            {m.structured.emergency_criteria.map((c, j) => (
                              <li key={j}>{c}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {m.structured.key_questions?.length > 0 && (
                        <div className="rounded-lg border border-gray-200 bg-white p-2 text-xs md:text-sm">
                          <p className="mb-1 font-semibold text-gray-700">감별을 위한 핵심 질문</p>
                          <ul className="list-inside list-disc space-y-0.5 text-gray-600">
                            {m.structured.key_questions.map((q, j) => (
                              <li key={j}>{q}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {m.structured.recommended_categories?.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          <span className="w-full text-xs font-semibold text-gray-600">이어서 물어보기</span>
                          {m.structured.recommended_categories.map((cat, j) => (
                            <button
                              key={j}
                              type="button"
                              onClick={() => sendWithQuery(cat.query)}
                              disabled={loading || !available}
                              className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 hover:bg-indigo-100 disabled:opacity-50 md:text-sm touch-manipulation"
                            >
                              {cat.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="mb-3 flex justify-start">
                  <div className="max-w-[90%] rounded-2xl bg-indigo-50 px-3 py-2.5 text-sm text-indigo-700 md:max-w-[85%] md:rounded-lg">
                    {waitingSeconds < 60
                      ? `답변 생성 중… ${waitingSeconds}초`
                      : `답변 생성 중… ${Math.floor(waitingSeconds / 60)}분 ${waitingSeconds % 60}초 (최대 10분까지 기다려 주세요)`}
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
            <div className="border-t border-gray-200 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] md:p-3 md:pb-3">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSend();
                }}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={!available || loading}
                  placeholder={available ? "메시지 입력..." : "LLM 사용 불가"}
                  className="min-w-0 flex-1 rounded-xl border border-gray-300 px-4 py-3 text-base focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:opacity-60 md:rounded-lg md:py-2 md:text-sm [font-size:16px]"
                />
                <button
                  type="submit"
                  disabled={!available || loading || !input.trim()}
                  className="shrink-0 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-medium text-white hover:bg-indigo-700 active:bg-indigo-800 disabled:opacity-50 md:rounded-lg md:py-2 touch-manipulation"
                >
                  전송
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
