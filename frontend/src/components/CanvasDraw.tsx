"use client";

import { useRef, useEffect, useState } from "react";

export default function CanvasDraw({ onImageReady }: { onImageReady: (img: string) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDraw = (e: any) => {
    setDrawing(true);
    draw(e);
  };

  const endDraw = () => {
    setDrawing(false);
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.beginPath();
  };

  const draw = (e: any) => {
    if (!drawing) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
  };

  const canvas = canvasRef.current!;
  const dataUrl = canvas ? canvas.toDataURL("image/png") : "";
  onImageReady(dataUrl);

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={500}
        height={100}
        onMouseDown={startDraw}
        onMouseUp={endDraw}
        onMouseMove={draw}
        className="border"
      />
      <button onClick={clearCanvas} className="px-4 py-1 bg-black text-white rounded hover:cursor-pointer">
        Xoá
      </button>
    </div>
  );
}