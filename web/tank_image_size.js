import { app } from "../../scripts/app.js";
import { isChinese, showImageSizeHelp } from "./image_size_help.js";

function localizeTitle(node) {
    const titles = ["TankImageSize", "尺寸（比例 · 百万像素）· Tank", "Image Size (Ratio · Megapixels) · Tank"];
    if (titles.includes(node.title)) node.title = isChinese() ? titles[1] : titles[2];
}

function localizePreview(widget) {
    widget.label = isChinese() ? "尺寸预览" : "Size Preview";
    widget.tooltip = isChinese()
        ? "对齐后的宽 × 高和实际 MP。修改比例、像素面积或倍数时自动更新；此框只读。"
        : "Aligned width × height and actual MP. Updates when ratio, area or alignment changes. Read-only.";
}

function sizePreview(node) {
    const value = name => node.widgets?.find(w => w.name === name)?.value;
    const ratio = String(value("aspect_ratio") || "3:4");
    const mp = Number(value("megapixels"));
    const multiple = Number(value("multiple"));
    if (ratio === "跟随图像1") {
        const mpText = Number.isFinite(mp) && mp > 0 ? String(mp) : "—";
        const mText = Number.isFinite(multiple) && multiple > 0 ? String(multiple) : "—";
        return isChinese()
            ? `跟随输入图像1比例 · ${mpText} MP · ${mText} 对齐`
            : `Follow image 1 ratio · ${mpText} MP · align ${mText}`;
    }
    const [rw, rh] = ratio.split(":").map(Number);
    const area = mp * 1024 ** 2;
    if (!(rw > 0 && rh > 0 && area > 0 && multiple > 0)) return "—";
    const width = Math.max(multiple, Math.floor(Math.sqrt(area * rw / rh) / multiple + 0.5) * multiple);
    const height = Math.max(multiple, Math.floor(Math.sqrt(area * rh / rw) / multiple + 0.5) * multiple);
    return `${width} × ${height}  (${(width * height / 1024 ** 2).toFixed(2)} MP)`;
}

app.registerExtension({
    name: "Tank.ImageSize",
    setup() {
        app.ui.settings.addEventListener("Comfy.Locale.change", () => {
            for (const node of app.graph?._nodes || []) {
                if (node.comfyClass !== "TankImageSize") continue;
                localizeTitle(node);
                const widget = node.widgets?.find(w => w.name === "size_preview");
                if (widget) {
                    localizePreview(widget);
                    widget.value = sizePreview(node);
                }
            }
            const dialog = document.querySelector('[data-tank-size-help]');
            if (dialog) { dialog.close(); showImageSizeHelp(); }
            app.graph?.setDirtyCanvas(true, true);
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TankImageSize") return;
        const drawTitle = nodeType.prototype.onDrawTitle;
        nodeType.prototype.onDrawTitle = function (ctx) {
            const result = drawTitle?.apply(this, arguments);
            if (this.flags?.collapsed) return result;
            const x = this.size[0] - 22;
            const y = -(LiteGraph.NODE_TITLE_HEIGHT || 30) / 2;
            ctx.save();
            ctx.fillStyle = "#414650";
            ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "#f2f3f5";
            ctx.font = "bold 14px sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("?", x, y + 0.5); ctx.restore();
            return result;
        };
        const mouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, pos) {
            const x = this.size[0] - 22;
            const y = -(LiteGraph.NODE_TITLE_HEIGHT || 30) / 2;
            if (!this.flags?.collapsed && event.button === 0 && pos && Math.hypot(pos[0] - x, pos[1] - y) <= 12) {
                showImageSizeHelp();
                return true;
            }
            return mouseDown?.apply(this, arguments);
        };
        const menu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = menu?.apply(this, arguments);
            options.push({ content: isChinese() ? "推荐尺寸说明 · Tank" : "Recommended Image Sizes · Tank", callback: showImageSizeHelp });
            return result;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const node = this;
            const preview = this.addWidget("text", "size_preview", "", () => {}, { readonly: true });
            localizePreview(preview);
            preview.serialize = false;
            preview.inputEl?.setAttribute("readonly", "readonly");
            const update = () => {
                localizeTitle(node);
                preview.value = sizePreview(node);
                node.setDirtyCanvas(true, true);
            };
            for (const widget of this.widgets || []) {
                if (!["aspect_ratio", "megapixels", "multiple"].includes(widget.name)) continue;
                const callback = widget.callback;
                widget.callback = function () {
                    const output = callback?.apply(this, arguments);
                    update();
                    return output;
                };
            }
            const configured = this.onConfigure;
            this.onConfigure = function () {
                const output = configured?.apply(this, arguments);
                update();
                return output;
            };
            update();
            this.size = [Math.max(this.size[0], 380), this.computeSize()[1]];
            return result;
        };
    },
});
