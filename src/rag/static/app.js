const form = document.querySelector("#recommend-form");
const queryInput = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const loadingPanel = document.querySelector("#loading");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const results = document.querySelector("#results");
const answer = document.querySelector("#answer");
const dishGrid = document.querySelector("#dish-grid");
const resetButton = document.querySelector("#reset-button");
const retryButton = document.querySelector("#retry-button");
const loadingTitle = document.querySelector("#loading-title");
const loadingMessage = document.querySelector("#loading-message");

const loadingSteps = [
  ["正在理解需求", "提取口味、忌口、天气和预算偏好..."],
  ["正在检索本地菜单", "筛掉不合适的菜品，保留更接近的一餐..."],
  ["正在组织推荐理由", "把菜品事实整理成更好读的建议..."],
];

let currentQuery = "";
let loadingTimer = null;

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    queryInput.focus();
  });
});

resetButton.addEventListener("click", () => {
  results.hidden = true;
  queryInput.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

retryButton.addEventListener("click", () => {
  if (!currentQuery) return;
  submitRecommendation(currentQuery);
});

function tag(text) {
  const item = document.createElement("span");
  item.textContent = text;
  return item;
}

function stat(label, value) {
  const item = document.createElement("div");
  item.className = "dish-stat";
  const title = document.createElement("span");
  title.textContent = label;
  const body = document.createElement("strong");
  body.textContent = value || "-";
  item.append(title, body);
  return item;
}

function compactAnswer(text) {
  return (text || "")
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 4)
    .join("\n");
}

function buildMatchReason(metadata, query) {
  const reasons = [];
  const compactQuery = query.replace(/\s/g, "");
  const isNoSpicy = /不辣|不要辣|免辣|不能吃辣/.test(compactQuery);
  const wantsSoup = /汤|带汤|热乎|热腾腾|下雨/.test(compactQuery);
  const wantsLight = /清淡|减脂|低卡|健康/.test(compactQuery);
  const wantsFull = /饱腹|管饱|晚餐|主食/.test(compactQuery);

  if (isNoSpicy && Number(metadata.spicy) === 0) reasons.push("不辣");
  if (wantsSoup && (`${metadata.temperature || ""}${metadata.tags || ""}${metadata.category || ""}`).includes("热")) {
    reasons.push("热食");
  }
  if (wantsSoup && (`${metadata.name || ""}${metadata.category || ""}${metadata.tags || ""}`).includes("汤")) {
    reasons.push("带汤");
  }
  if (wantsLight && (`${metadata.calorie_level || ""}${metadata.scene || ""}${metadata.tags || ""}`).match(/低|减脂|清淡|健康/)) {
    reasons.push("清淡友好");
  }
  if (wantsFull && Number(metadata.satiety || 0) >= 4) reasons.push("饱腹感强");
  if (!reasons.length && metadata.taste) reasons.push(`${metadata.taste}口味`);
  if (!reasons.length && metadata.scene) reasons.push(`适合${metadata.scene.split(/\s+/)[0]}`);
  return reasons.slice(0, 3).join("、") || "与当前需求接近";
}

function createDishCard(dish) {
  const metadata = dish.metadata || {};
  const card = document.createElement("article");
  card.className = "dish-card";

  const top = document.createElement("div");
  top.className = "dish-top";
  const name = document.createElement("h4");
  name.textContent = metadata.name || "未命名菜品";
  const price = document.createElement("span");
  price.className = "dish-price";
  price.textContent = metadata.price ? `¥${metadata.price}` : "价格未知";
  top.append(name, price);

  const reason = document.createElement("p");
  reason.className = "dish-reason";
  reason.textContent = buildMatchReason(metadata, currentQuery);

  const stats = document.createElement("div");
  stats.className = "dish-stats";
  stats.append(
    stat("热量", metadata.calorie_kcal ? `${metadata.calorie_kcal} kcal` : "未知"),
    stat("辣度", Number(metadata.spicy || 0) === 0 ? "不辣" : `${metadata.spicy}/5`),
    stat("饱腹", metadata.satiety ? `${metadata.satiety}/5` : "未知"),
    stat("评分", metadata.rating ? `${metadata.rating}` : "未知")
  );

  const meta = document.createElement("div");
  meta.className = "dish-meta";
  [metadata.category, metadata.taste, metadata.temperature && `${metadata.temperature}食`, metadata.scene]
    .filter(Boolean)
    .forEach((value) => meta.appendChild(tag(value)));

  const match = document.createElement("div");
  match.className = "match-row";
  const track = document.createElement("div");
  track.className = "match-track";
  const fill = document.createElement("div");
  fill.className = "match-fill";
  const percent = Math.max(0, Math.min(100, Math.round((dish.score || 0) * 100)));
  fill.style.width = `${percent}%`;
  track.appendChild(fill);
  const label = document.createElement("small");
  label.textContent = `匹配 ${percent}%`;
  match.append(track, label);

  card.append(top, reason, stats, meta, match);
  return card;
}

function startLoading() {
  let index = 0;
  const render = () => {
    const [title, message] = loadingSteps[Math.min(index, loadingSteps.length - 1)];
    loadingTitle.textContent = title;
    loadingMessage.textContent = message;
    index += 1;
  };
  render();
  clearInterval(loadingTimer);
  loadingTimer = setInterval(render, 4500);
}

function stopLoading() {
  clearInterval(loadingTimer);
  loadingTimer = null;
}

async function submitRecommendation(query) {
  currentQuery = query;
  submitButton.disabled = true;
  loadingPanel.hidden = false;
  errorPanel.hidden = true;
  results.hidden = true;
  startLoading();

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "推荐服务暂时不可用");

    answer.textContent = compactAnswer(data.answer);
    dishGrid.replaceChildren();
    (data.retrieved_dishes || []).forEach((dish) => dishGrid.appendChild(createDishCard(dish)));
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    errorMessage.textContent = error.message || "请稍后再试";
    errorPanel.hidden = false;
  } finally {
    stopLoading();
    loadingPanel.hidden = true;
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  submitRecommendation(query);
});
