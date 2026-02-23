import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  addTrainingItem,
  deleteTrainingItem,
  getTrainingItems,
  getTrainingCategories,
  getTrainingImageFullUrl,
  getErrorMessage,
  startTraining,
  updateTrainingCaption,
  updateTrainingItem,
} from "../services/api";
import type { TrainingItem } from "../types/api";

export default function TrainingPage() {
  const [items, setItems] = useState<TrainingItem[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [addFile, setAddFile] = useState<File | null>(null);
  const [addCaption, setAddCaption] = useState("");
  const [addCategory, setAddCategory] = useState("");
  const [adding, setAdding] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainCategory, setTrainCategory] = useState<string>("");
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editCaption, setEditCaption] = useState("");
  const [editCategory, setEditCategory] = useState("");

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const list = await getTrainingItems();
      setItems(list);
    } catch (err) {
      setMessage({ type: "error", text: getErrorMessage(err) });
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchCategories = useCallback(async () => {
    try {
      const list = await getTrainingCategories();
      setCategories(list);
    } catch {
      setCategories([]);
    }
  }, []);

  useEffect(() => {
    fetchItems();
    fetchCategories();
  }, [fetchItems, fetchCategories]);

  const handleAdd = useCallback(async () => {
    if (!addFile) return;
    setAdding(true);
    setMessage(null);
    try {
      await addTrainingItem(addFile, addCaption, addCategory || undefined);
      setAddFile(null);
      setAddCaption("");
      setAddCategory("");
      await fetchItems();
      await fetchCategories();
      setMessage({ type: "success", text: "추가되었습니다." });
    } catch (err) {
      setMessage({ type: "error", text: getErrorMessage(err) });
    } finally {
      setAdding(false);
    }
  }, [addFile, addCaption, addCategory, fetchItems, fetchCategories]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteTrainingItem(id);
        await fetchItems();
      } catch (err) {
        setMessage({ type: "error", text: getErrorMessage(err) });
      }
    },
    [fetchItems]
  );

  const handleStartEdit = useCallback((item: TrainingItem) => {
    setEditingId(item.id);
    setEditCaption(item.caption);
    setEditCategory(item.category ?? "");
  }, []);

  const handleSaveEdit = useCallback(
    async (id: string) => {
      try {
        await updateTrainingItem(id, { caption: editCaption, category: editCategory || undefined });
        setEditingId(null);
        await fetchItems();
        await fetchCategories();
      } catch (err) {
        setMessage({ type: "error", text: getErrorMessage(err) });
      }
    },
    [editCaption, editCategory, fetchItems, fetchCategories]
  );

  const handleStartTraining = useCallback(async () => {
    setTraining(true);
    setMessage(null);
    try {
      const categoryToUse = trainCategory.trim() || null;
      const result = await startTraining(categoryToUse);
      if (result.error) {
        setMessage({ type: "error", text: result.message ?? result.error });
      } else {
        setMessage({ type: "success", text: result.message ?? "학습을 시작했습니다." });
      }
    } catch (err) {
      setMessage({ type: "error", text: getErrorMessage(err) });
    } finally {
      setTraining(false);
    }
  }, [trainCategory]);

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="mx-auto max-w-4xl px-4">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">LoRA 학습</h1>
            <p className="mt-2 text-gray-600">이미지와 프롬프트 라벨을 넣고 학습 데이터를 만든 뒤 학습을 시작하세요.</p>
          </div>
          <div className="flex gap-2">
            <Link
              to="/"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              이미지 생성
            </Link>
            <Link
              to="/chat"
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              LLM 채팅
            </Link>
          </div>
        </header>

        <div className="space-y-6">
          {/* 추가 폼 */}
          <section className="rounded-xl bg-white p-6 shadow">
            <h2 className="mb-4 text-sm font-semibold text-gray-700">학습 데이터 추가</h2>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
                <div className="flex-1">
                  <label className="mb-1 block text-xs text-gray-500">이미지 (AI 생성 이미지 또는 업로드)</label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setAddFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-sm text-gray-500 file:mr-2 file:rounded file:border-0 file:bg-indigo-50 file:px-3 file:py-2 file:text-indigo-700"
                  />
                </div>
                <div className="flex-1">
                  <label className="mb-1 block text-xs text-gray-500">프롬프트 라벨 (캡션)</label>
                  <input
                    type="text"
                    value={addCaption}
                    onChange={(e) => setAddCaption(e.target.value)}
                    placeholder="예: a black cat on white quilt, two cats"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                </div>
                <div className="flex-1">
                  <label className="mb-1 block text-xs text-gray-500">카테고리 (학습 시 이 카테고리로 묶음)</label>
                  <input
                    type="text"
                    value={addCategory}
                    onChange={(e) => setAddCategory(e.target.value)}
                    placeholder="예: 픽셀아트, anime"
                    list="category-list"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                  {categories.length > 0 && (
                    <datalist id="category-list">
                      {categories.map((c) => (
                        <option key={c} value={c} />
                      ))}
                    </datalist>
                  )}
                </div>
                <button
                  type="button"
                  onClick={handleAdd}
                  disabled={!addFile || adding}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                >
                  {adding ? "추가 중…" : "추가"}
                </button>
              </div>
            </div>
          </section>

          {message && (
            <div
              role="alert"
              className={`rounded-lg p-4 text-sm ${
                message.type === "error" ? "border border-red-200 bg-red-50 text-red-800" : "border border-green-200 bg-green-50 text-green-800"
              }`}
            >
              {message.text}
            </div>
          )}

          {/* 목록 */}
          <section className="rounded-xl bg-white p-6 shadow">
            <h2 className="mb-4 text-sm font-semibold text-gray-700">학습 데이터 ({items.length}건)</h2>
            {loading ? (
              <p className="text-sm text-gray-500">불러오는 중…</p>
            ) : items.length === 0 ? (
              <p className="text-sm text-gray-500">아직 데이터가 없습니다. 위에서 이미지와 캡션을 추가하세요.</p>
            ) : (
              <ul className="space-y-4">
                {items.map((item) => (
                  <li key={item.id} className="flex flex-col gap-3 rounded-lg border border-gray-200 p-3 sm:flex-row sm:items-start">
                    <img
                      src={getTrainingImageFullUrl(item.image_url)}
                      alt=""
                      className="h-24 w-24 shrink-0 rounded object-cover"
                    />
                    <div className="min-w-0 flex-1">
                      {editingId === item.id ? (
                        <div className="space-y-2">
                          <input
                            type="text"
                            value={editCaption}
                            onChange={(e) => setEditCaption(e.target.value)}
                            placeholder="캡션"
                            className="w-full rounded border border-gray-300 px-2 py-1 text-sm"
                          />
                          <input
                            type="text"
                            value={editCategory}
                            onChange={(e) => setEditCategory(e.target.value)}
                            placeholder="카테고리"
                            className="w-full rounded border border-gray-300 px-2 py-1 text-sm"
                          />
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={() => handleSaveEdit(item.id)}
                              className="rounded bg-indigo-600 px-2 py-1 text-xs text-white hover:bg-indigo-700"
                            >
                              저장
                            </button>
                            <button
                              type="button"
                              onClick={() => setEditingId(null)}
                              className="rounded bg-gray-200 px-2 py-1 text-xs hover:bg-gray-300"
                            >
                              취소
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          {item.category && (
                            <span className="inline-block rounded bg-gray-200 px-1.5 py-0.5 text-xs text-gray-700">
                              {item.category}
                            </span>
                          )}
                          <p className="mt-1 text-sm text-gray-700">{item.caption || "(캡션 없음)"}</p>
                          <button
                            type="button"
                            onClick={() => handleStartEdit(item)}
                            className="mt-1 text-xs text-indigo-600 hover:underline"
                          >
                            수정
                          </button>
                        </>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleDelete(item.id)}
                      className="rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700 hover:bg-red-100 sm:shrink-0"
                    >
                      삭제
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* 학습 시작 */}
          <section className="rounded-xl bg-white p-6 shadow">
            <h2 className="mb-2 text-sm font-semibold text-gray-700">LoRA 학습 시작</h2>
            <p className="mb-3 text-xs text-gray-500">
              카테고리를 선택하면 해당 카테고리에 속한 사진만 사용해 학습합니다. 서버에서 백그라운드로 실행되며 완료까지 시간이 걸릴 수 있습니다.
            </p>
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <label className="text-sm text-gray-600">학습할 카테고리:</label>
              <select
                value={trainCategory}
                onChange={(e) => setTrainCategory(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              >
                <option value="">전체 (모든 데이터)</option>
                {categories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <span className="text-xs text-gray-500">
                {trainCategory.trim()
                  ? `"${trainCategory}" 카테고리만 학습`
                  : "전체 데이터로 학습"}
              </span>
            </div>
            <button
              type="button"
              onClick={handleStartTraining}
              disabled={items.length === 0 || training}
              className="rounded-lg bg-emerald-600 px-4 py-3 font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
            >
              {training ? "시작 중…" : "학습 시작"}
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}
