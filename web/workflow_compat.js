import { app } from "../../scripts/app.js";
import { installWorkflowCompatibility } from "./workflow_compat.mjs";

app.registerExtension({
  name: "TankNodes.WorkflowCompatibility",
  beforeRegisterNodeDef(nodeType, nodeData) {
    installWorkflowCompatibility(nodeType, nodeData.name);
  },
});
