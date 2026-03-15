# About
Okhra is a fast and accurate AI detector that you can run locally on a CPU. It uses a convolutional neural network to capture stylistic choices characteristic of LLMs.

# How to install
You can use Okhra as a web extension for Firefox\Chrome or as a CLI tool.

For Chrome browser, you can download a zip file from the releases and then drag-an-drop it into the browser window.
Alternatively:
```
git clone https://github.com/okhra-dev/okhra
```
Then go to `chrome://extensions/`, select *"Load unpacked"* and select the `okhra` folder

For Firefox:
```
git clone https://github.com/okhra-dev/okhra
```
Then go to `about:debugging` in your browser, select `This Firefox`, then `Load Temporary Add-on...`. Navigate to the `okhra` folder and select `manifest.json`

Okhra will be available to install from the web store as soon as it passes the review process.

# How to use

You can either select any text on the page, then right-click it and choose `Check with Okhra` or click the extension icon and paste the text in the box. Note that predictions may not be accurate for small texts (less than 100 words). You can select a false positive rate from 0.1% to 1% and toggle dark theme.

# Accuracy

On our private evaluation dataset the model achieves 99.3% TPR at FPR = 1% and 95.5% TPR at FPR = 0.1%. This dataset covers several domains (news articles, creative writing, social media posts) and a wide variety of models. Note that on some texts (e.g. wikipedia articles) the false positive rate may be higher.
