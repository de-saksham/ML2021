
javascript:(function() {
    // Fetch the i18n JSON file - this could also be done dynamically like
    // window.location.origin - would always return the domain name
    fetch('https://test-argo-alb.goodyear.eu/en_gb/consumer.i18nlist.json')
      .then(response => response.json())
      .then(data => {
        // Get all text nodes on the web page
        const textNodes = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false); // creates a TreeWalker object that traverses the DOM tree and selects only the text nodes.
        // Iterate through each text node
        while (textNodes.nextNode()) {
          const node = textNodes.currentNode;

          if(node && node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent.trim();
            // Check if the text matches a translation key
            if (Object.values(data).includes(text)) {
                const translationKey = Object.keys(data).find(key => data[key] === text);
                // Replace the text node with the translation key
                node.textContent = translationKey;
            }

            // Check if the text matches a translation key
            if (data.hasOwnProperty(text)) {
              // Replace the text node with the translation text
              node.textContent = data[text];
            }
          }
        }
      })
      .catch(error => {
        console.error('Error fetching i18n JSON:', error);
      });
  })();