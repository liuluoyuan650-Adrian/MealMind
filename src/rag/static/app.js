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

function tag(text) {
  const item = document.createElement("span");
  item.textContent = text;
  return item;
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

  const meta = document.createElement("div");
  meta.className = "dish-meta";
  [metadata.category, metadata.taste, metadata.temperature && `${metadata.temperature}食`,
    metadata.calorie_kcal && `${metadata.calorie_kcal} kcal`]
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

  card.append(top, meta, match);
  return card;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  submitButton.disabled = true;
  loadingPanel.hidden = false;
  errorPanel.hidden = true;
  results.hidden = true;

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "推荐服务暂时不可用");

    answer.textContent = data.answer;
    dishGrid.replaceChildren();
    (data.retrieved_dishes || []).forEach((dish) => dishGrid.appendChild(createDishCard(dish)));
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    errorMessage.textContent = error.message || "请稍后再试";
    errorPanel.hidden = false;
  } finally {
    loadingPanel.hidden = true;
    submitButton.disabled = false;
  }
});
