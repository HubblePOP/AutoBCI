import re

with open('dashboard/index.html', 'r') as f:
    content = f.read()

# Define the boundaries of the right column
start_marker = '      <section class="mission-right-column">'
end_marker = '      <div class="secondary-masonry">'

if start_marker in content and end_marker in content:
    before = content.split(start_marker)[0]
    after = content.split(end_marker)[1]
    
    new_right_col = """      <section class="mission-right-column">
        <section id="task-console-panel" class="panel console-card">
          <div class="console-head">
            <div>
              <div class="console-title">任务终端 Console</div>
              <div class="console-copy">当前任务、关键问题、下一步和控制按钮。</div>
            </div>
            <div class="mission-chip-row">
              <span class="mission-chip off" id="planner-status-chip">Planner</span>
            </div>
          </div>
          <div class="console-field">
            <div class="console-field-label">Current Task</div>
            <div id="current-task" class="console-field-value">-</div>
          </div>
          <div class="console-field">
            <div class="console-field-label">Current Problem</div>
            <div id="current-problem" class="console-field-value">-</div>
          </div>
          <div class="console-field">
            <div class="console-field-label">Next Step</div>
            <div id="next-step" class="console-field-value">-</div>
          </div>
          <label class="console-field" for="task-input">
            <div class="console-field-label">Task Input</div>
            <textarea id="task-input" class="console-field-value" rows="4" style="resize: vertical; border: 0; background: transparent; padding: 0; font: inherit; color: inherit;" placeholder="输入一条研究任务，例如：继续做同试次纯脑电 moonshot，优先 formal 前 2 名。"></textarea>
          </label>
          <div class="console-actions" id="control-buttons">
            <button class="console-button" type="button" data-action="think">Think</button>
            <button class="console-button" type="button" data-action="execute">Execute</button>
            <button class="console-button" type="button" data-action="pause">Pause</button>
            <button class="console-button" type="button" data-action="resume">Resume</button>
            <button class="console-button" type="button" data-action="end">End</button>
          </div>
          <div id="control-response" class="mission-list-empty">控制台会把动作结果显示在这里。</div>
        </section>

        <section class="panel run-card">
          <div class="run-card-head">
            <div class="run-card-title">
              <strong>当前运行任务 Current Run</strong>
              <span id="hero-effect">-</span>
            </div>
            <div class="run-card-note" id="hero-sub">只看数据集、方法、阶段、时间、时长和最近一次正式效果。</div>
          </div>
          <div class="hero-effect-note">
            <strong>当前正式效果</strong>
            <span id="hero-formal-note">-</span>
            <div id="hero-effect-source" class="hero-formal-summary">-</div>
          </div>
          <div class="hero-summary-grid run-card-grid">
            <div class="hero-summary-row"><span class="hero-summary-label">数据集</span><span id="hero-dataset" class="hero-summary-value">-</span></div>
            <div class="hero-summary-row"><span class="hero-summary-label">方法</span><span id="hero-method" class="hero-summary-value">-</span></div>
            <div class="hero-summary-row"><span class="hero-summary-label">阶段</span><span id="hero-stage" class="hero-summary-value">-</span></div>
            <div class="hero-summary-row"><span class="hero-summary-label">最近更新</span><span id="hero-time" class="hero-summary-value">-</span></div>
            <div class="hero-summary-row"><span class="hero-summary-label">已运行时长</span><span id="hero-duration" class="hero-summary-value">-</span></div>
          </div>
        </section>
      </section>
      </div>

      <div class="mission-full-width" style="display: grid; gap: 18px; margin-top: 18px; margin-bottom: 18px;">
        <section class="panel mission-stack-card">
          <div class="mission-stack-head">
            <div class="mission-stack-title">核心指标演进 SOTA Evolution</div>
            <div id="rmse-coverage" class="mission-stack-note">不仅是数字的堆砌，更是研究思路的进化史。</div>
          </div>
          <div class="mainline-trend-copy" style="margin-bottom: 16px;">
            记录每一次模型突破与背后的探索代价。<strong>大节点</strong>代表 SOTA (历史最佳) 的刷新，<strong>暗淡的小点</strong>代表未成功的尝试。
          </div>

          <!-- 占据全宽的主图：相关系数 r 的演进，重点展示 -->
          <article class="chart-card epic-trend-slot" style="margin-bottom: 20px;">
            <div class="chart-head">
              <div class="chart-title">
                <div class="chart-label">核心指标 (相关系数 r)</div>
                <div class="chart-value">主线 SOTA 演进轨迹</div>
              </div>
              <div class="chart-legend">
                <span><i class="kept milestone"></i>SOTA 突破点 (成功采纳)</span>
                <span><i class="discarded cloud"></i>探索尝试 (未留用)</span>
                <span><i class="running line"></i>当前实验主线</span>
              </div>
            </div>
            <div id="mainline-primary" class="chart-shell">
              <div class="summary-empty">这张图当前还没有真实点。</div>
            </div>
          </article>

          <!-- 辅助图表：RMSE等，放在下方网格 -->
          <div class="mainline-trend-grid">
            <article class="chart-card mainline-trend-slot">
              <div class="chart-head">
                <div class="chart-title">
                  <div class="chart-label">验证 RMSE</div>
                  <div class="chart-value">误差收敛趋势</div>
                </div>
                <div class="chart-legend">
                  <span><i class="running"></i>深色线：当前正式主线</span>
                </div>
              </div>
              <div id="mainline-rmse" class="chart-shell">
                <div class="summary-empty">这张图当前还没有真实点。</div>
              </div>
            </article>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">最佳模型榜单 Scoreboard</div>
            <div id="mainline-note" class="panel-note">这张图不依赖当前有没有在跑，会一直保留主线历史。</div>
          </div>
          <div id="mainline-family-best-strip" class="moonshot-strip">
            <div class="moonshot-empty">当前还没有可显示的候选模型。</div>
          </div>
        </section>
      </div>

      <div class="secondary-masonry">
"""

    with open('dashboard/index.html', 'w') as f:
        f.write(before + new_right_col + after)
    print("Layout replaced successfully.")
else:
    print("Markers not found.")

