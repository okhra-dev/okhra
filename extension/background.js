chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "okhra-analyze",
    title: "Analyze with Okhra",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === "okhra-analyze" && info.selectionText) {
    chrome.storage.local.set({
      pendingText: info.selectionText,
      pendingTs: Date.now()
    });
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#C8963E" });
  }
});
