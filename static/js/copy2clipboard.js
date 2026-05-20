function copyBibtex() {
  const bibtexText = document.getElementById("bibtexContent").innerText;
  const button = document.getElementById("copyButton");

  navigator.clipboard.writeText(bibtexText).then(function () {
    button.innerText = "Copied";
    setTimeout(function () {
      button.innerText = "Copy";
    }, 2000);
  });
}
