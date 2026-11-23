import { app } from "../../scripts/app.js";

export function isChinese() {
    return String(app.ui.settings.getSettingValue("Comfy.Locale") || "en").toLowerCase().startsWith("zh");
}

const recommendations = [
    ["H3 Image", "4–20M；约 16M 效果最佳", "省资源 2048 × 2048（4M）；高清 4096 × 4096 / 5472 × 3072（约 16M）"],
    ["Qwen2.1", "1–4M；默认 2M", "1024 × 1024 ～ 2048 × 2048；默认方图约 1440 × 1440"],
    ["MOSS", "1–2.5M；默认 1M", "1024 × 1024 ～ 1632 × 1632；默认 1024 × 1024"],
    ["FLUX3 / Flux.2", "1–2M；默认 1M", "1024 × 1024 ～ 1440 × 1440"],
    ["Z-Image / Ideogram4 / Krea2", "1–2M", "1024 × 1024 ～ 1440 × 1440"],
    ["其他旧图像模型", "1–2M", "方图 1024²～1440²；横图 1344 × 768～1920 × 1088"],
];

const recommendationsEn = [
    ["H3 Image", "4–20M; best around 16M", "Fast: 2048 × 2048 (4M); detailed: 4096 × 4096 / 5472 × 3072 (~16M)"],
    ["Qwen2.1", "1–4M; default 2M", "1024 × 1024 to 2048 × 2048; default square ~1440 × 1440"],
    ["MOSS", "1–2.5M; default 1M", "1024 × 1024 to 1632 × 1632; default 1024 × 1024"],
    ["FLUX3 / Flux.2", "1–2M; default 1M", "1024 × 1024 to 1440 × 1440"],
    ["Z-Image / Ideogram4 / Krea2", "1–2M", "1024 × 1024 to 1440 × 1440"],
    ["Other older image models", "1–2M", "Square: 1024²–1440²; landscape: 1344 × 768–1920 × 1088"],
];

function element(tag, text, style) {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (style) Object.assign(el.style, style);
    return el;
}

export function showImageSizeHelp() {
    const zh = isChinese();
    const existing = document.querySelector('[data-tank-size-help]');
    if (existing) { existing.focus(); return; }
    const dialog = element("dialog", null, {
        width: "min(790px, 92vw)", maxHeight: "85vh", overflow: "auto", padding: "24px",
        background: "#25272c", color: "#eeeeef", border: "1px solid #60636c", borderRadius: "12px",
        font: "14px/1.65 sans-serif", boxSizing: "border-box",
    });
    dialog.dataset.tankSizeHelp = "true";
    const header = element("div", null, { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "20px" });
    header.append(element("h2", zh ? "常用出图尺寸" : "Recommended Image Sizes", { margin: "0", fontSize: "20px" }));
    const close = element("button", zh ? "关闭" : "Close", { padding: "5px 15px", cursor: "pointer" });
    close.addEventListener("click", () => dialog.close());
    header.append(close); dialog.append(header);
    const scroll = element("div", null, { overflowX: "auto", marginTop: "16px" });
    const table = element("table", null, { width: "100%", borderCollapse: "collapse", textAlign: "left" });
    const head = element("tr");
    for (const text of zh ? ["模型", "建议面积", "常用分辨率（宽 × 高）"] : ["Model", "Suggested Area", "Common Resolution (W × H)"]) {
        head.append(element("th", text, { padding: "10px", borderBottom: "1px solid #656973" }));
    }
    const thead = element("thead"); thead.append(head); table.append(thead);
    const tbody = element("tbody");
    for (const row of zh ? recommendations : recommendationsEn) {
        const tr = element("tr");
        for (const text of row) tr.append(element("td", text, { padding: "10px", borderBottom: "1px solid #41444c", verticalAlign: "top" }));
        tbody.append(tr);
    }
    table.append(tbody); scroll.append(table); dialog.append(scroll);
    dialog.append(element("p", zh ? "1M = 1024² 像素。超出推荐范围可能影响画质、增加显存占用或导致生成失败。" : "1M = 1024² pixels. Exceeding the recommended range may reduce quality, increase VRAM use or cause generation to fail.", { margin: "14px 0 0", color: "#c5c8cf" }));
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    document.body.append(dialog); dialog.showModal();
}
