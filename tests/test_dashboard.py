"""Run dashboard state logic with a tiny DOM and controlled delayed responses."""
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "node is required for the dashboard state test")
class DashboardTests(unittest.TestCase):
    def test_delayed_search_and_info_do_not_regress_generation(self):
        dashboard = Path(__file__).resolve().parents[1] / "evidence_lab" / "dashboard.html"
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
let js = html.split('<script>')[1].split('</script>')[0].replace(/openScope\(\);\s*$/, '');
class Element {
  constructor(id = '') { this.id=id; this.textContent=''; this.value=''; this.children=[]; this.handlers={}; this.classList={toggle(){}}; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children=[]; }
  addEventListener(type, fn) { this.handlers[type]=fn; }
}
const elements = {};
const document = {getElementById(id) { return elements[id] ||= new Element(id); }, createElement(tag) { return new Element(tag); }};
let resolveFetch;
const fetch = () => new Promise(resolve => { resolveFetch = resolve; });
const ctx = vm.createContext({document, fetch, console});
vm.runInContext(js, ctx);
vm.runInContext("selected = {tenant: 'local', collection: 'research'}; updateInfo({generation: 1, document_count: 1});", ctx);
document.getElementById('query').value='query';
(async () => {
  const pending = elements['search-form'].handlers.submit({preventDefault(){}});
  vm.runInContext("advanceGeneration(2); updateInfo({generation: 2, document_count: 2});", ctx);
  resolveFetch({ok:true, json:async () => ({generation:1, document_count:1, elapsed_ms:1, results:[{title:'stale', text:'old', parent_id:'x', score:1, start_char:0, end_char:3, passage_id:'p'}]})});
  await pending;
  assert.strictEqual(elements.generation.textContent, '2');
  assert.strictEqual(elements['doc-count'].textContent, '2');
  assert.ok(!JSON.stringify(elements.results.children).includes('stale'));
  vm.runInContext("updateInfo({generation: 1, document_count: 1})", ctx);
  assert.strictEqual(elements.generation.textContent, '2');
  assert.strictEqual(elements['doc-count'].textContent, '2');
})().catch(error => { console.error(error); process.exitCode=1; });
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(dashboard)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
