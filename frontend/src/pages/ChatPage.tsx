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
import type { ChatRoomSummary } from "../services/api";

type Message = { role: "user" | "assistant"; content: string };

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
  const bottomRef = useRef<HTMLDivElement>(null);

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
    if (!text || !available || loading) return;
    setInput("");
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    let roomId = currentRoomId;
    try {
      if (!roomId) {
        const room = await createChatRoom();
        roomId = room.id;
        setCurrentRoomId(roomId);
        await loadRooms();
      }
      await addChatMessage(roomId!, "user", text);
      const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }));
      const content = await llmChat(history, 1024, 0.4);
      const assistantMsg: Message = { role: "assistant", content };
      setMessages((prev) => [...prev, assistantMsg]);
      if (roomId) await addChatMessage(roomId, "assistant", content);
      await loadRooms();
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `오류: ${getErrorMessage(err)}` }]);
    } finally {
      setLoading(false);
    }
  }, [input, available, loading, messages, currentRoomId, loadRooms]);

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

  return (
    <div className="min-h-screen bg-gray-100 py-6">
      <div className="mx-auto flex max-w-5xl gap-4 px-4">
        {/* 채팅방 목록 */}
        <aside className="w-52 shrink-0 rounded-xl bg-white shadow">
          <div className="border-b border-gray-200 p-3">
            <button
              type="button"
              onClick={startNewChat}
              className="w-full rounded-lg bg-indigo-600 py-2 text-sm font-medium text-white hover:bg-indigo-700"
            >
              새 채팅
            </button>
          </div>
          <div className="max-h-[calc(100vh-12rem)] overflow-y-auto p-2">
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
                  onClick={() => loadRoom(r.id)}
                  onKeyDown={(e) => e.key === "Enter" && loadRoom(r.id)}
                  className={`group flex items-center justify-between gap-1 rounded-lg px-2 py-2 text-left text-sm hover:bg-gray-50 ${
                    currentRoomId === r.id ? "bg-indigo-50 text-indigo-700" : "text-gray-700"
                  }`}
                >
                  <span className="min-w-0 flex-1 truncate" title={r.title}>
                    {r.title}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteRoom(e, r.id)}
                    className="shrink-0 rounded p-1 text-gray-400 opacity-0 hover:bg-gray-200 hover:text-red-600 group-hover:opacity-100"
                    aria-label="삭제"
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        <div className="min-w-0 flex-1">
          <header className="mb-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">건강 질문 도우미</h1>
              <p className="mt-1 text-sm text-gray-500">
                {available
                  ? "증상·반려동물 상황을 적어 주세요. 참고 정보만 제공하며, 정확한 판단은 의료·수의 전문가에게 확인하세요."
                  : "LLM을 사용할 수 없습니다."}
              </p>
            </div>
            <Link
              to="/"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              이미지 생성
            </Link>
          </header>

          <div className="rounded-xl bg-white shadow">
            <div className="flex h-[60vh] flex-col overflow-y-auto p-4">
              {messages.length === 0 && !loading && (
                <p className="text-center text-sm text-gray-500">
                  증상이나 반려동물 상황을 간단히 적어 주세요. (참고 정보만 제공합니다)
                </p>
              )}
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`mb-3 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                      m.role === "user"
                        ? "bg-indigo-600 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {m.role === "assistant" ? renderAssistantContent(m.content) : m.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="mb-3 flex justify-start">
                  <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm text-gray-500">
                    입력 중…
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
            <div className="border-t border-gray-200 p-3">
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
                  className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-60"
                />
                <button
                  type="submit"
                  disabled={!available || loading || !input.trim()}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
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
