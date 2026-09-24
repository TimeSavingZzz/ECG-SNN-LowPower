#!/usr/bin/env python3
"""答辩现场**零依赖**离线演示包：把两个前端打成单文件 HTML。

## 为什么必须做（不是锦上添花，有实测依据）

答辩现场的实时演示路径是「本机浏览器 → 腾讯云 → 跳板机 → GPU → 容器」，
三层 SSH 嵌套。实测（2026-09-24，同一样本 ``ecg_id=9``，响应 331411 B）：

===================  ==========  ==========
位置                  耗时         吞吐
===================  ==========  ==========
容器内自连（urllib）   **1.76 s**   188 KB/s
经隧道（本机 curl）    **47.6 s**   6.9 KB/s
===================  ==========  ==========

⇒ **推理本身只要 1.76 秒，慢出来的 27 倍全在隧道搬字节。**
而响应体积的 99.2% 是 ``ecg``(12×1000 浮点，占 76.9%) 与
``input_spikes``(8032 个坐标，占 22.3%) 两项；gzip 只压得动 4.6:1
（浮点文本高熵），救不回来。

⇒ 结论：**答辩不该走这条链路。** 本脚本把面板 HTML/CSS/JS 与预置样本的
推理结果全部内联进一个文件，拷到本机后 ``file://`` 双击即开：
零网络、零服务、点样本**瞬时**切换。会场网络故障也不影响。

## 怎么做到「不改前端一行逻辑」

在 ``app.js`` **之前**注入一个 ``window.fetch`` 拦截器，按 URL 从内联的
``window.__BUNDLE__`` 取数据并返回 ``new Response(...)``。前端的 ``jget()``
与 ``r.json()`` 完全察觉不到差别；上游 ``app.js`` 保持只读底本。

## 两个面板的差异（决定它们走不同分支）

* **PTB-XL**：数据靠 5 个**动态**接口，必须预跑推理。``--collect`` 连容器内
  的 ``10_serve_demo.py``（shim），把 5 个路由 + ``/examples`` 里**每一个**
  样本的 ``/infer_example`` 全收下来。
* **MIT-BIH**：本来就是纯静态（``fetch("data/*.json")`` + 相对路径资源），
  扫 ``app.js`` 里的字面量路径逐个内联即可，缺数据会**当场报错**而不是
  静默少一块。

## 用法::

    # 容器内（需 10_serve_demo.py 已在 8000 端口运行）
    PY=/opt/miniconda3/envs/shadocformer/bin/python
    $PY experiments/13_build_offline_demo.py --all

    # bundle 已在时只重打包，不连 shim
    $PY experiments/13_build_offline_demo.py --build

产物（``results/`` 已被 .gitignore，不污染仓库）::

    results/offline/ptbxl_bundle.json   原始 bundle（可复算）
    results/offline/ptbxl_demo.html     ← 拷回本机，双击打开
    results/offline/mitbih_demo.html

## 学术诚信

``ecg`` 逐点四舍五入到 4 位小数（``--ecg-decimals``，设 0 则不降精度）。
信号范围约 ±3 ⇒ 4 位小数即 1e-4 分辨率，而屏幕一个像素约 2e-3
⇒ **可视化上不可见**，只为把单文件从 ~6.7 MB 压到 ~3.5 MB 以便回传。
数值口径（``probs`` / ``spike_rates`` / ``compare_probs`` / ``macs`` /
``sops`` / ``macro_auroc``）**一个字节都不动**，生成时会打印实际压缩量。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")

# 前端请求的 URL 里可能带版本号/查询串，匹配时统统剥掉再看路径。
_CSS_LINK = re.compile(r'<link\b[^>]*href\s*=\s*"([^"]*\.css[^"]*)"[^>]*>', re.I)
# 指向外网的 <link>（上游 index.html:7-9 的 Google Fonts 三条）
_REMOTE_LINK = re.compile(r'<link\b[^>]*href\s*=\s*"https?://[^"]*"[^>]*>', re.I)
_JS_SRC = re.compile(r'<script\b[^>]*src\s*=\s*"([^"]+)"[^>]*>\s*</script>', re.I)
# 只扫字面量 fetch("...") / fetch('...')，够用且不会误伤模板拼接的 URL。
_FETCH_LIT = re.compile(r"""fetch\(\s*["'`]([^"'`]+)["'`]\s*\)""")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 收集（PTB-XL：动态接口，必须预跑推理）
# ---------------------------------------------------------------------------
def _get_json(base: str, path: str, timeout: int = 1800) -> dict:
    """请求 shim 的一个路由，连 HTTP 状态码一起带回来（404 也是有效信息）。"""
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return {"status": r.status, "body": json.loads(r.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"detail": raw[:400]}
        return {"status": e.code, "body": body}


def _round_floats(a, nd: int):
    """把嵌套 list 里的浮点降到 nd 位（只为缩小体积，见模块 docstring）。"""
    if isinstance(a, list):
        return [_round_floats(v, nd) for v in a]
    if isinstance(a, float):
        return round(a, nd)
    return a


def collect_ptbxl(base: str, out: Path, n_examples: int, fold: int,
                  ecg_decimals: int) -> dict:
    log(f"[collect] shim = {base}")
    endpoints = {}
    # 前端实际发的请求：/examples 带 n 与 fold，/history 带 run_name=main
    # （见 10_serve_demo.py 的 _history：它把 main 映射到我们的 repro_snn）。
    for path in ("/status", f"/examples?n={n_examples}&fold={fold}",
                 "/history?run_name=main", "/comparison"):
        key = path.split("?")[0]
        r = _get_json(base, path)
        endpoints[key] = r
        n = len(json.dumps(r["body"])) if r["body"] is not None else 0
        log(f"  {key:<12} HTTP {r['status']}  {n} B")

    ex = endpoints.get("/examples", {}).get("body") or {}
    ids = [int(e["ecg_id"]) for e in ex.get("examples", [])]
    if not ids:
        log("[collect] ⚠️ /examples 为空 —— 前端 LABELS 的唯一来源就是它，"
            "离线包会整页空白。先检查 shim 的 --cache-dir。")
        return {"endpoints": endpoints, "infer": {}}

    log(f"[collect] 预跑 {len(ids)} 个样本的推理（容器内约 2 s/个）...")
    infer = {}
    t0 = time.time()
    for i, eid in enumerate(ids, 1):
        r = _get_json(base, f"/infer_example?ecg_id={eid}")
        if r["status"] == 200 and ecg_decimals > 0:
            body = r["body"]
            if isinstance(body.get("ecg"), list):
                body["ecg"] = _round_floats(body["ecg"], ecg_decimals)
            if isinstance(body.get("vout"), list):
                body["vout"] = _round_floats(body["vout"], ecg_decimals)
            r["body"] = body
        infer[str(eid)] = r
        log(f"  [{i}/{len(ids)}] ecg_id={eid}  HTTP {r['status']}  "
            f"{len(json.dumps(r['body']))} B  ({time.time()-t0:.0f}s)")

    bundle = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": base,
        "examples_fold": fold,
        "ecg_decimals": ecg_decimals,
        "endpoints": endpoints,
        "infer": infer,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    log(f"[collect] 写出 {out}  {out.stat().st_size/1e6:.2f} MB")
    return bundle


# ---------------------------------------------------------------------------
# 构建（两个面板共用）
# ---------------------------------------------------------------------------
def _interceptor_js(bundle: dict) -> str:
    """生成 fetch 拦截器。它必须在 app.js **之前**执行。"""
    blob = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)
    # JSON 里出现 </script> 会提前闭合宿主标签（数据里不会有，但内联脚本必须防）
    blob = blob.replace("</", "<\\/")
    return (
        "<script>\n"
        "/* 13_build_offline_demo.py 注入：离线演示包 */\n"
        "window.__BUNDLE__ = " + blob + ";\n"
        "(function () {\n"
        "  var B = window.__BUNDLE__ || {};\n"
        "  var EP = B.endpoints || {}, INF = B.infer || {}, ST = B.static || {};\n"
        "  function resp(body, status) {\n"
        "    return new Response(JSON.stringify(body === undefined ? null : body), {\n"
        "      status: status || 200,\n"
        "      headers: { 'Content-Type': 'application/json; charset=utf-8' }\n"
        "    });\n"
        "  }\n"
        "  window.fetch = function (input, init) {\n"
        "    var url = (typeof input === 'string') ? input\n"
        "            : (input && input.url) ? input.url : String(input);\n"
        "    var method = ((init && init.method) || (input && input.method)\n"
        "                  || 'GET').toUpperCase();\n"
        "    var path = url.split('?')[0].replace(/^[a-z]+:\\/\\/[^\\/]*/i, '');\n"
        "    if (path.length > 1 && path.charAt(path.length - 1) === '/')\n"
        "      path = path.slice(0, -1);\n"
        "    if (method === 'POST') {\n"
        "      return Promise.resolve(resp({ detail:\n"
        "        '离线演示包不含上传推理（需联网跑后端）。现场演示请用样本选择器。' }, 503));\n"
        "    }\n"
        "    for (var k in EP) {\n"
        "      if (EP.hasOwnProperty(k) && path === k)\n"
        "        return Promise.resolve(resp(EP[k].body, EP[k].status));\n"
        "    }\n"
        "    if (path === '/infer_example') {\n"
        "      var m = /[?&]ecg_id=(\\d+)/.exec(url);\n"
        "      var hit = m && INF[m[1]];\n"
        "      if (hit) return Promise.resolve(resp(hit.body, hit.status));\n"
        "      return Promise.resolve(resp({ detail:\n"
        "        '离线包未预置该样本 ecg_id=' + (m ? m[1] : '?') }, 404));\n"
        "    }\n"
        "    for (var s in ST) {\n"
        "      if (ST.hasOwnProperty(s) && (path === s || path === '/' + s\n"
        "          || path.slice(-(s.length + 1)) === '/' + s))\n"
        "        return Promise.resolve(resp(ST[s], 200));\n"
        "    }\n"
        "    return Promise.resolve(resp({ detail:\n"
        "      'offline demo: no data for ' + path }, 404));\n"
        "  };\n"
        "})();\n"
        "</script>\n"
    )


def build_one(panel: str, html_path: Path, out_path: Path, bundle: dict):
    """把一个面板的 index.html 就地内联成单文件。

    ``extra_files`` 是「相对引用 → 文件内容」映射，用于 CSS。
    JS 专门处理：拦截器必须在它前面注入。
    """
    if not html_path.is_file():
        log(f"[build] ⚠️ 跳过 {panel}：找不到 {html_path}")
        return None
    html = html_path.read_text(encoding="utf-8")
    base = html_path.parent
    inline_js: list[str] = []
    inlined: list[str] = []
    missing: list[str] = []

    # 0) 先剔除指向外网的 <link>（上游 index.html:7-9 的 Google Fonts 三条）。
    #    本包要的是「零依赖、双击即开」：现场若**有网但慢**，浏览器会为了等
    #    fonts.googleapis.com 而白屏数秒；无网则静默回退。而本地化时已把
    #    Microsoft YaHei / PingFang SC / Noto Sans CJK 补进 CSS 的 --font 回退链，
    #    故删外链只影响字形美观，不影响任何正确性。
    dropped = _REMOTE_LINK.findall(html)
    if dropped:
        html = _REMOTE_LINK.sub("", html)
        log(f"  [build] 剔除 {len(dropped)} 条外网 <link>（离线包不依赖网络）")

    # 1) CSS
    def _css(m):
        href = m.group(1)
        p = base / href.split("?")[0].lstrip("/").replace("static/", "", 1)
        if not p.is_file():
            p = base / href.split("?")[0].lstrip("/")
        if p.is_file():
            inlined.append(p.name)
            return "<style>\n" + p.read_text(encoding="utf-8") + "\n</style>"
        missing.append(href)
        return m.group(0)

    html = _CSS_LINK.sub(_css, html)

    # 2) JS：先收敛成列表，统一在最后按顺序拼（拦截器永远排第一）
    def _js(m):
        src = m.group(1)
        p = base / src.split("?")[0].lstrip("/").replace("static/", "", 1)
        if not p.is_file():
            p = base / src.split("?")[0].lstrip("/")
        if p.is_file():
            inline_js.append(p.read_text(encoding="utf-8"))
            inlined.append(p.name)
            return "<!--JS:%s-->" % p.name
        missing.append(src)
        return m.group(0)

    html = _JS_SRC.sub(_js, html)

    # 3) 扫 JS 里的字面量 fetch("...")，逐个内联静态数据（MIT-BIH 走这条）
    static_data: dict = {}
    for js in list(inline_js):
        for rel in _FETCH_LIT.findall(js):
            if rel.startswith(("http://", "https://")):
                continue
            p = base / rel.split("?")[0]
            if p.is_file() and p.suffix == ".json":
                try:
                    static_data[rel.split("?")[0]] = json.loads(
                        p.read_text(encoding="utf-8"))
                except Exception as e:
                    log(f"[build] ⚠️ {panel}: {rel} 解析失败 {e}")
    for rel, content in static_data.items():
        log(f"  [build] 内联静态数据 {rel}  "
            f"{len(json.dumps(content))/1e6:.2f} MB")
    if static_data:
        bundle = dict(bundle)
        bundle["static"] = static_data

    # 4) 拦截器 + 各 JS。
    #    顺序要紧：拦截器必须排在 app.js **之前**，否则 app.js 会拿原生 fetch
    #    去请求 /status 等绝对路径 —— file:// 下那是文件系统路径，必然失败。
    parts = [_interceptor_js(bundle)]
    for js_code in inline_js:
        parts.append("<script>\n" + js_code + "\n</script>\n")
    combined = "\n".join(parts)
    placeholders = re.findall(r"<!--JS:(.*?)-->", html)
    if placeholders:
        # 全部 JS 已合并在第一个占位处，其余占位清空（保持文档结构不变）
        html = html.replace("<!--JS:%s-->" % placeholders[0], combined, 1)
        for name in placeholders[1:]:
            html = html.replace("<!--JS:%s-->" % name, "", 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    size = out_path.stat().st_size
    log(f"[build] {panel} → {out_path}  {size/1e6:.2f} MB")
    log(f"        内联: {', '.join(dict.fromkeys(inlined)) or '(无)'}")
    if missing:
        log(f"[build] ⚠️ 未找到外部引用（离线打开会缺资源）: {missing}")

    # 5) 自检分两类：<script src>/<link href> 是**硬泄漏**必须为 0；
    #    而 app.js 源码里的 fetch("data/x.json") 字面量是**预期内**的 ——
    #    我们不改前端源码，而是让拦截器在运行时截获，故只需核对去向已被覆盖。
    txt = out_path.read_text(encoding="utf-8")
    hard = []
    for pat in (r'src\s*=\s*"(?!data:)[^"]*\.js"',
                r'href\s*=\s*"(?!data:)[^"]*\.css"',
                # 只有**资源加载类**标签的外链才算硬泄漏；正文里的
                # <a href="https://physionet.org/...">（数据出处引用）必须保留。
                r'<(?:link|script|img|iframe|source|video|audio)\b[^>]*'
                r'\b(?:src|href)\s*=\s*"https?://[^"]*"'):
        hard += [m2.group(0)[:70] for m2 in re.finditer(pat, txt)]

    covered, uncovered = [], []
    for m2 in re.finditer(r'fetch\(\s*["\']([^"\']+)["\']', txt):
        rel = m2.group(1)
        if rel.startswith(("http://", "https://")):
            continue
        if rel.split("?")[0] in static_data:
            covered.append(rel)        # 已内联成 bundle.static
        elif rel.startswith("/"):
            covered.append(rel)        # 绝对路由，由 endpoints/infer 覆盖
        else:
            uncovered.append(rel)

    if hard or uncovered:
        log(f"[build] ⚠️ 自检发现 {len(hard) + len(uncovered)} 处**真实**外部引用:")
        for s in (hard + [f'fetch("{r}")' for r in uncovered])[:8]:
            log(f"          {s}")
    else:
        log(f"[build] ✅ 自检通过：无残留外部资源引用（{len(covered)} 处 fetch "
            f"均由内联数据或拦截器覆盖），可 file:// 直接打开")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="答辩现场离线演示包（单文件 HTML）")
    ap.add_argument("--all", action="store_true", help="收集 + 打包两个面板")
    ap.add_argument("--collect", action="store_true", help="只连 shim 收 PTB-XL 数据")
    ap.add_argument("--build", action="store_true", help="只用已有 bundle 打包")
    ap.add_argument("--panel", choices=["ptbxl", "mitbih", "both"], default="both")
    ap.add_argument("--shim", default="http://127.0.0.1:8000")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "offline"))
    ap.add_argument("--web-ptbxl", default=str(ROOT / "results" / "web_cn"))
    ap.add_argument("--web-mitbih", default=str(ROOT / "results" / "mitbih" / "web_cn"))
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--fold", type=int, default=10)
    ap.add_argument("--ecg-decimals", type=int, default=4,
                    help="ecg 保留小数位（0=不降精度）；见模块 docstring")
    args = ap.parse_args()

    if not (args.all or args.collect or args.build):
        args.all = True
    out_dir = Path(args.out_dir)
    bundle_path = out_dir / "ptbxl_bundle.json"

    need_ptbxl = args.panel in ("ptbxl", "both")
    need_mitbih = args.panel in ("mitbih", "both")

    if need_ptbxl and (args.all or args.collect):
        bundle = collect_ptbxl(args.shim, bundle_path, args.n_examples,
                               args.fold, args.ecg_decimals)
    elif need_ptbxl:
        if not bundle_path.is_file():
            log(f"[build] 缺少 {bundle_path} —— 先跑 --collect")
            sys.exit(1)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        log(f"[build] 读入 bundle {bundle_path} "
            f"({bundle_path.stat().st_size/1e6:.2f} MB，"
            f"生成于 {bundle.get('generated_at')})")
    else:
        bundle = {}

    if need_ptbxl:
        build_one("PTB-XL", Path(args.web_ptbxl) / "index.html",
                  out_dir / "ptbxl_demo.html", bundle)
    if need_mitbih:
        build_one("MIT-BIH", Path(args.web_mitbih) / "index.html",
                  out_dir / "mitbih_demo.html", {})  # 纯静态，无需 bundle

    log("[done] 把 results/offline/*.html 拷回本机，浏览器双击打开即可演示")


if __name__ == "__main__":
    main()
