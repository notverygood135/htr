"use client";

import CanvasDraw from "@/components/CanvasDraw";
import { useState } from "react";

export default function Home() {
  const [result, setResult] = useState("");
  const [canvasUrl, setCanvasUrl] = useState("");

  const handleRecognition = async () => {
    if (!canvasUrl) return;
    const res = await fetch("http://localhost:8000/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvasUrl }),
    });
    const data = await res.json();
    setResult(data.text);
  };

  return (
    <div className="h-screen p-4 bg-white">
      <h1 className="text-xl mb-2 text-black">Nhận diện chữ viết tay</h1>
      <CanvasDraw onImageReady={setCanvasUrl} />
      <button
        onClick={handleRecognition}
        className="mt-2 px-4 py-2 bg-black text-white rounded hover:cursor-pointer"
      >
        Nhận diện
      </button>
      {result && <p className="mt-4 text-black">Kết quả: <b>{result}</b></p>}
    </div>
  );
}
