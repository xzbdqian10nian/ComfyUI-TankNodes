// Transform only known historical widget layouts. Never reorder node widgets.
const API_NODES = new Set(["VisionAPIEnv", "VisionAPIDirect"]);
const LEGACY_REASONING = {
  backend_default: "auto", thinking: "medium", instruct: "off",
  none: "off", disabled: "off",
};
const SEED_CONTROLS = new Set(["fixed", "increment", "decrement", "randomize"]);
const VIDEO_TRANSPORTS = new Set(["auto", "frames", "video_url"]);
const LAYOUT_PROPERTY = "tanknodes_widget_layout";

export function migrateWidgetValues(type, values) {
  if (!Array.isArray(values)) return values;
  const isApi = API_NODES.has(type);
  if (!isApi && type !== "VisionChat") return values;
  let migrated = values;
  // v0.5 API: seed control, max_video_frames, video_transport. v0.6 added
  // thinking_mode before the optional video widgets. Detect the full shape.
  if (isApi && values.length === 11 && SEED_CONTROLS.has(values[8]) &&
      typeof values[9] === "number" && VIDEO_TRANSPORTS.has(values[10])) {
    migrated = [...values.slice(0, 9), "auto", ...values.slice(9)];
  }
  const index = isApi ? 9 : 2;
  const value = migrated[index];
  const replacement = typeof value === "string" ? LEGACY_REASONING[value] : undefined;
  if (replacement) {
    migrated = migrated.slice();
    migrated[index] = replacement;
  }
  return migrated;
}

export function installWorkflowCompatibility(nodeType, type) {
  if (!API_NODES.has(type) && type !== "VisionChat") return;
  const configure = nodeType.prototype.configure;
  nodeType.prototype.configure = function (data, ...rest) {
    let values = data?.widgets_values;
    const isApi = API_NODES.has(type);
    const legacyShape = Array.isArray(values) && (isApi
      ? values.length === 11 && SEED_CONTROLS.has(values[8]) && VIDEO_TRANSPORTS.has(values[10])
      : values.length === 16 && ["backend_default", "thinking", "instruct"].includes(values[2]));
    const names = (data?.inputs ?? []).map(input => input.name);
    const promptIndex = names.indexOf("prompt");
    const systemIndex = names.indexOf("system_prompt");
    // v0.5.0 used prompt before system_prompt. Saved input metadata can
    // identify that order without guessing from the user's text. Keep input
    // slots untouched so links remain valid; mark the normalized widget order.
    if (legacyShape && data?.properties?.[LAYOUT_PROPERTY] !== 1 &&
        data?.properties?.qwen38_vl_prompt_order !== "system_first" &&
        promptIndex >= 0 && systemIndex > promptIndex) {
      const index = isApi ? 3 : 0;
      if (typeof values[index] === "string" && typeof values[index + 1] === "string") {
        values = values.slice();
        [values[index], values[index + 1]] = [values[index + 1], values[index]];
      }
    }
    values = migrateWidgetValues(type, values);
    const next = values === data?.widgets_values ? data : { ...data, widgets_values: values };
    const result = configure.call(this, next, ...rest);
    this.properties = { ...this.properties, [LAYOUT_PROPERTY]: 1 };
    return result;
  };
}
