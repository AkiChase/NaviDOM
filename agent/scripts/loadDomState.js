(frameViewport = null) => {
    if (frameViewport === null) {
        frameViewport = { offsetX: 0, offsetY: 0, left: 0, top: 0, bottom: Infinity, right: Infinity };
    }

    function loadDomState(root) {
        function getAttributes(el) {
            const attrs = {};
            for (const attr of el.attributes || []) {
                attrs[attr.name] = attr.value;
            }
            return attrs;
        }

        function getClientBounds(el) {
            if (typeof el.getBoundingClientRect !== "function") {
                return null;
            }
            const rect = el.getBoundingClientRect();
            const visible = !(
                rect.bottom <= 0 ||
                rect.right <= 0 ||
                rect.top >= viewportHeight ||
                rect.left >= viewportWidth ||
                rect.width <= 0 ||
                rect.height <= 0
            );
            if (!visible) return null;

            const actualRect = {
                top: rect.top + frameViewport.offsetY,
                left: rect.left + frameViewport.offsetX,
                bottom: rect.bottom + frameViewport.offsetY,
                right: rect.right + frameViewport.offsetX,
                width: rect.width,
                height: rect.height,
            }
            const actualVisible = !(
                actualRect.bottom <= frameViewport.top ||
                actualRect.right <= frameViewport.left ||
                actualRect.top >= frameViewport.bottom ||
                actualRect.left >= frameViewport.right
            );
            if (!actualVisible) return null;

            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const topEl = document.elementFromPoint(cx, cy);
            const isCovered = topEl !== el && !el.contains(topEl);
            // works not well for iframe

            const out = {
                x: actualRect.left,
                y: actualRect.top,
                width: actualRect.width,
                height: actualRect.height,
                isCovered,
            }

            if (el.tagName.toLowerCase() === "iframe") {
                const style = window.getComputedStyle(el);
                const paddingLeft = parseFloat(style.paddingLeft);
                const paddingTop = parseFloat(style.paddingTop);
                out.offsetX = el.clientLeft + paddingLeft;
                out.offsetY = el.clientTop + paddingTop;
            }

            return out;
        }

        function isStyleVisible(el) {
            const style = window.getComputedStyle(el);
            if (style.display === "none") return false;
            if (style.visibility === "hidden" || style.visibility === "collapse") return false;
            if (style.opacity === "0") return false;
            if (el.hasAttribute("hidden")) return false;
            return true;
        }

        function getElementChildIndex(el) {
            const parent = el.parentElement;
            if (!parent) return null;
            return Array.prototype.indexOf.call(parent.children, el) + 1;
        }

        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const stack = [{ el: root, visited: false }];
        const nodeMap = new Map();

        while (stack.length) {
            const { el, visited } = stack.pop();

            if (!visited) {
                stack.push({ el, visited: true });
                if (el.tagName === "IFRAME") continue; // skip iframe children

                // const children = Array.from(el.childNodes);
                const children = [];
                children.push(...el.childNodes);
                // Shadow DOM children
                if (el.shadowRoot) {
                    children.push(...el.shadowRoot.childNodes);
                }
                for (let i = children.length - 1; i >= 0; i--) {
                    const child = children[i];
                    if (child.nodeType !== Node.ELEMENT_NODE && child.nodeType !== Node.TEXT_NODE) continue; // skip non-element/text node
                    if (child.tagName === "SCRIPT" || child.tagName === "STYLE") continue; // skip script/style node
                    if (child.nodeType === Node.TEXT_NODE && child.textContent.trim() === "") continue; // skip empty text node
                    stack.push({ el: child, visited: false });
                }
            } else {
                // collect final children
                const children = [];
                for (const child of el.childNodes) {
                    const childNode = nodeMap.get(child);
                    if (childNode) {
                        children.push(childNode);
                    }
                }
                for (const child of el.shadowRoot?.childNodes || []) {
                    const childNode = nodeMap.get(child);
                    if (childNode) {
                        children.push(childNode);
                    }
                }

                if (el.nodeType === Node.TEXT_NODE) {
                    // text node
                    const node = {
                        tag: "",
                        attrs: {},
                        bounds: null,
                        children,
                        text: el.textContent.trim(),
                        index: null,
                        selector: null,
                    };
                    nodeMap.set(el, node);
                } else {
                    const curBounds = getClientBounds(el);
                    const tag = el.tagName.toLowerCase();
                    let selector = null
                    if (!!el.closest('select')) {
                        // Do not consider child nodes for <select/> 
                        if (tag === 'select') {
                            if ((curBounds === null || !isStyleVisible(el))) continue;
                            selector = CssSelectorGenerator.getCssSelector(el, { includeTag: true, maxCombinations: 10 })
                        }
                        // keep all child nodes of <select/> with null selector
                    } else {
                        // skip invisible elements without visible children
                        if ((children.length === 0 || children.every(child => child.tag === "")) && (curBounds === null || !isStyleVisible(el))) continue;
                        selector = CssSelectorGenerator.getCssSelector(el, { includeTag: true, maxCombinations: 10 })
                    }
                    const node = {
                        tag,
                        attrs: getAttributes(el),
                        bounds: curBounds,
                        children,
                        text: "",
                        index: getElementChildIndex(el),
                        selector,
                    };
                    nodeMap.set(el, node);
                }
            }
        }

        return {
            dom: nodeMap.get(root),
            viewport: {
                width: viewportWidth,
                height: viewportHeight,
                scrollY: window.scrollY,
                scrollX: window.scrollX,
                scrollH: document.documentElement.scrollHeight,
                scrollW: document.documentElement.scrollWidth,
            },
        };
    }
    return loadDomState(document.body);
}