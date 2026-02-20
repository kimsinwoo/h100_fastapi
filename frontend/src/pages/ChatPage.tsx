import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { getLlmStatus, llmChat, getErrorMessage } from "../services/api";

type Message = { role: "user" | "assistant"; content: string };

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [available, setAvailable] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getLlmStatus()
      .then((r) => {
        setAvailable(r.available);
        setModel(r.model ?? null);
      })
      .catch(() => setAvailable(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || !available || loading) return;
    setInput("");
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const history = [...messages, userMsg].map((m) => ({ role: m.role, content: m.content }));
      const content = await llmChat(history, 512, 0.7);
      setMessages((prev) => [...prev, { role: "assistant", content }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `오류: ${getErrorMessage(err)}` }]);
    } finally {
      setLoading(false);
    }
  }, [input, available, loading, messages]);

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="mx-auto max-w-3xl px-4">
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">LLM 채팅 (gpt-oss-20b)</h1>
            <p className="mt-1 text-sm text-gray-500">
              {available ? `모델: ${model ?? "gpt-oss-20b"}` : "LLM 서버를 사용할 수 없습니다."}
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
              <p className="text-center text-sm text-gray-500">메시지를 입력하고 전송하세요.</p>
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
                  {m.content}
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
  );
}
