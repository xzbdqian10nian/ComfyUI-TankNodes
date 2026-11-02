// 提示词模板 · Tank: a "管理模板" button on the node opens a small manager to view, edit, copy, add and delete templates.
// Built-in templates are read-only (copy them to make your own). Saving refreshes the dropdown on every template node.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const T = {
  zh: { button: "管理模板（编辑 / 复制 / 新增）", title: "提示词模板", name: "模板名称", builtin: "内置模板（只读，可复制后修改）",
        mine: "我的模板", save: "保存", copy: "复制为新模板", add: "新建空白模板", del: "删除", close: "关闭", use: "用这个模板",
        copySuffix: "（副本）", newName: "我的新模板", confirmDel: "确定删除这个模板？", saved: "已保存", needName: "请填写模板名称" },
  en: { button: "Manage templates (edit / copy / add)", title: "Prompt templates", name: "Template name", builtin: "Built-in (read-only, copy to edit)",
        mine: "My template", save: "Save", copy: "Copy as new", add: "New blank", del: "Delete", close: "Close", use: "Use this template",
        copySuffix: " (copy)", newName: "My new template", confirmDel: "Delete this template?", saved: "Saved", needName: "Please enter a name" },
};
function L() {
  let loc = "zh";
  try { loc = app.ui.settings.getSettingValue("Comfy.Locale") || navigator.language || "zh"; } catch { /* default */ }
  return String(loc).toLowerCase().startsWith("zh") ? T.zh : T.en;
}

async function fetchTemplates() {
  const r = await api.fetchApi("/tank/prompt_templates");
  return (await r.json()).templates || [];
}
async function post(path, body) {
  const r = await api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
  return data.templates || [];
}

// Push the new label list into every template node's dropdown (and optionally select a label on one node).
function refreshNodes(templates, node, select) {
  const labels = templates.map(t => t.label);
  for (const n of app.graph?._nodes || []) {
    if (n.comfyClass !== "TankPromptTemplate" && n.type !== "TankPromptTemplate") continue;
    const w = n.widgets?.find(x => x.name === "template");
    if (!w) continue;
    w.options.values = labels;
    if (!labels.includes(w.value)) w.value = labels[0];
  }
  if (node && select) {
    const w = node.widgets?.find(x => x.name === "template");
    if (w) { w.value = select; w.callback?.(select); }
  }
  app.graph?.setDirtyCanvas(true, true);
}

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "style") Object.assign(e.style, v); else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v); }
  for (const k of [].concat(kids)) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

async function openManager(node) {
  const t = L();
  let templates = await fetchTemplates();
  const cur = node.widgets?.find(x => x.name === "template")?.value;
  let sel = templates.find(x => x.label === cur) || templates[0];
  let editingNew = false;

  const btn = (label, onclick, primary) => el("button", { onclick, style: { padding: "6px 12px", borderRadius: "8px", border: "1px solid #555",
    background: primary ? "#3b82f6" : "#2a2a2e", color: "#eee", cursor: "pointer", fontSize: "13px" } }, label);
  const list = el("div", { style: { width: "260px", overflow: "auto", borderRight: "1px solid #333", paddingRight: "8px" } });
  const nameIn = el("input", { type: "text", style: { width: "100%", padding: "6px 8px", background: "#1e1e22", color: "#eee", border: "1px solid #444", borderRadius: "6px", boxSizing: "border-box" } });
  const badge = el("div", { style: { fontSize: "12px", color: "#9aa", margin: "6px 0" } });
  const text = el("textarea", { spellcheck: "false", style: { flex: "1", width: "100%", minHeight: "320px", padding: "8px", background: "#1e1e22", color: "#ddd",
    border: "1px solid #444", borderRadius: "6px", fontFamily: "ui-monospace,Consolas,monospace", fontSize: "12px", boxSizing: "border-box", resize: "none" } });
  const msg = el("span", { style: { fontSize: "12px", color: "#9c9", marginLeft: "8px" } });
  const delBtn = btn(t.del, async () => {
    if (!sel || sel.builtin || editingNew || !confirm(t.confirmDel)) return;
    try { templates = await post("/tank/prompt_templates/delete", { label: sel.label }); sel = templates[0]; refreshNodes(templates); paint(); }
    catch (e) { msg.style.color = "#e88"; msg.textContent = e.message; }
  });
  const saveBtn = btn(t.save, async () => {
    const label = nameIn.value.trim(); if (!label) { msg.style.color = "#e88"; msg.textContent = t.needName; return; }
    try {
      templates = await post("/tank/prompt_templates", { label, text: text.value, old_label: (!editingNew && sel && !sel.builtin) ? sel.label : "" });
      sel = templates.find(x => x.label === label); editingNew = false; refreshNodes(templates, node, label);
      msg.style.color = "#9c9"; msg.textContent = t.saved; paint();
    } catch (e) { msg.style.color = "#e88"; msg.textContent = e.message; }
  }, true);

  function paint() {
    list.replaceChildren(...templates.map(x => el("div", { onclick: () => { sel = x; editingNew = false; msg.textContent = ""; paint(); },
      style: { padding: "6px 8px", borderRadius: "6px", cursor: "pointer", fontSize: "13px", marginBottom: "2px",
        background: sel && x.label === sel.label && !editingNew ? "#33415c" : "transparent", color: x.builtin ? "#ccc" : "#bfe3ff" } },
      (x.builtin ? "" : "★ ") + x.label)));
    if (editingNew) return;
    nameIn.value = sel?.label || ""; text.value = sel?.text || "";
    const ro = !!sel?.builtin;
    nameIn.readOnly = ro; text.readOnly = ro;
    badge.textContent = ro ? t.builtin : t.mine;
    saveBtn.style.display = ro ? "none" : ""; delBtn.style.display = ro ? "none" : "";
  }
  function startNew(name, body) {
    editingNew = true; nameIn.readOnly = false; text.readOnly = false; nameIn.value = name; text.value = body;
    badge.textContent = t.mine; saveBtn.style.display = ""; delBtn.style.display = "none"; msg.textContent = ""; nameIn.focus(); paint();
  }

  const box = el("div", { style: { position: "fixed", inset: "0", background: "rgba(0,0,0,.55)", zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center" },
    onclick: e => { if (e.target === box) box.remove(); } }, [
    el("div", { style: { width: "min(1100px,94vw)", height: "min(760px,90vh)", background: "#18181b", color: "#eee", border: "1px solid #333",
      borderRadius: "12px", padding: "14px", display: "flex", flexDirection: "column", gap: "10px", boxShadow: "0 20px 60px rgba(0,0,0,.5)" } }, [
      el("div", { style: { fontSize: "15px", fontWeight: "600" } }, t.title),
      el("div", { style: { flex: "1", display: "flex", gap: "12px", minHeight: "0" } }, [
        list,
        el("div", { style: { flex: "1", display: "flex", flexDirection: "column", minWidth: "0" } },
          [el("label", { style: { fontSize: "12px", color: "#aaa", marginBottom: "4px" } }, t.name), nameIn, badge, text]),
      ]),
      el("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, [
        btn(t.use, () => { if (sel && !editingNew) refreshNodes(templates, node, sel.label); box.remove(); }, true),
        saveBtn,
        btn(t.copy, () => startNew((sel?.label || t.newName) + t.copySuffix, sel?.text || "")),
        btn(t.add, () => startNew(t.newName, "")),
        delBtn, msg, el("span", { style: { flex: "1" } }), btn(t.close, () => box.remove()),
      ]),
    ]),
  ]);
  document.body.append(box);
  paint();
}

app.registerExtension({
  name: "TankNodes.PromptTemplateManager",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "TankPromptTemplate") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const w = this.addWidget("button", L().button, null, () => openManager(this), { serialize: false });
      if (w) w.serialize = false;  // not part of widgets_values: keeps old workflows' template / extra_rules order intact
      return r;
    };
  },
  async setup() {
    // Templates added on another page / by another user show up after a reload; refresh once at start too.
    try { refreshNodes(await fetchTemplates()); } catch { /* ComfyUI still loading */ }
  },
});
