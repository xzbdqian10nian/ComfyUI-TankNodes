import { test } from 'node:test';
import assert from 'node:assert/strict';
import { migrateWidgetValues, installWorkflowCompatibility } from '../web/workflow_compat.mjs';

const oldApi = ['http://127.0.0.1/v1', 'model', 'test-key', 'persona', 'prompt', 0, 0, 1, 'fixed', 8, 'frames'];
for (const type of ['VisionAPIEnv', 'VisionAPIDirect']) {
  test(`${type}: migrate old API without touching saved strings or trailing parameters`, () => {
    const original = structuredClone(oldApi);
    const current = migrateWidgetValues(type, oldApi);
    assert.deepEqual(current.slice(0, 9), original.slice(0, 9));
    assert.deepEqual(current.slice(9), ['auto', 8, 'frames']);
    assert.deepEqual(oldApi, original);
    assert.equal(migrateWidgetValues(type, current), current);
  });
}
test('normalize local legacy reasoning only', () => {
  const values = ['persona', 'prompt', 'thinking', 8192];
  assert.deepEqual(migrateWidgetValues('VisionChat', values), ['persona', 'prompt', 'medium', 8192]);
  assert.equal(values[2], 'thinking');
});
test('leave unrecognized nodes and layouts intact', () => {
  assert.equal(migrateWidgetValues('OtherNode', oldApi), oldApi);
  const different = [...oldApi]; different[10] = 'unknown';
  assert.equal(migrateWidgetValues('VisionAPIDirect', different), different);
  assert.equal(migrateWidgetValues('VisionChat', undefined), undefined);
});
test('configure wrapper preserves this, title, links, other extensions and return value', () => {
  class Node {
    configure(data, extra) { this.received = data; this.extra = extra; return 123; }
  }
  installWorkflowCompatibility(Node, 'VisionAPIDirect');
  const node = new Node();
  const data = { title: 'My custom title', inputs: [{ link: 42 }], widgets_values: oldApi };
  assert.equal(node.configure(data, 'extra'), 123);
  assert.equal(node.extra, 'extra');
  assert.equal(node.received.title, data.title);
  assert.equal(node.received.inputs, data.inputs);
  assert.deepEqual(node.received.widgets_values.slice(9), ['auto', 8, 'frames']);
  assert.equal(data.widgets_values.length, 11);
});

for (const type of ['VisionAPIDirect', 'VisionChat']) {
  test(`${type}: migrate evidenced prompt-first layout once without moving input links`, () => {
    class Node { configure(data) { this.received = data; this.properties = data.properties; } }
    installWorkflowCompatibility(Node, type);
    const node = new Node();
    const values = type === 'VisionChat'
      ? ['user request', 'system persona', 'thinking', 8192, 4096, .6, .95, 40, .05, 1.05, 1, 'fixed', 8, 'frames', false, '']
      : [...oldApi.slice(0, 3), 'user request', 'system persona', ...oldApi.slice(5)];
    const inputs = [{ name: 'prompt', link: 41 }, { name: 'system_prompt', link: 42 }];
    const data = { inputs, properties: { custom: 'preserved' }, widgets_values: values };
    node.configure(data);
    const i = type === 'VisionChat' ? 0 : 3;
    assert.deepEqual(node.received.widgets_values.slice(i, i + 2), ['system persona', 'user request']);
    assert.equal(node.received.inputs, inputs);
    assert.equal(node.properties.custom, 'preserved');
    assert.equal(node.properties.tanknodes_widget_layout, 1);
    const saved = { ...node.received, properties: node.properties };
    node.configure(saved);
    assert.deepEqual(node.received.widgets_values, saved.widgets_values);
    assert.deepEqual(values.slice(i, i + 2), ['user request', 'system persona']);
  });
}
