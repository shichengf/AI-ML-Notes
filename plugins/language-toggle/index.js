import { h } from "preact"
import { resolveRelative } from "@quartz-community/utils"

function isEnglishSlug(slug) {
  return slug === "en/index" || slug.startsWith("en/")
}

function translatedSlug(slug) {
  if (isEnglishSlug(slug)) {
    return slug.slice(3) || "index"
  }

  return `en/${slug}`
}

function control(tag, text, properties = {}) {
  return h(tag, properties, text)
}

const styles = `
.language-toggle {
  display: inline-flex;
  margin-left: auto;
  padding: 0.2rem;
  gap: 0.15rem;
  border: 1px solid var(--lightgray);
  border-radius: 999px;
  background: color-mix(in srgb, var(--light) 88%, var(--lightgray));
  font-family: var(--bodyFont);
  font-size: 0.82rem;
  line-height: 1;
}

.language-toggle a,
.language-toggle span {
  box-sizing: border-box;
  min-width: 2.6rem;
  padding: 0.48rem 0.62rem;
  border-radius: 999px;
  text-align: center;
  white-space: nowrap;
}

.language-toggle a {
  color: var(--darkgray);
  transition: color 0.2s ease, background-color 0.2s ease;
}

.language-toggle a:hover,
.language-toggle a:focus-visible {
  color: var(--secondary);
  background: var(--highlight);
}

.language-toggle .is-active {
  color: var(--light);
  background: var(--secondary);
  font-weight: 700;
}

.language-toggle .is-disabled {
  color: var(--gray);
  cursor: not-allowed;
  opacity: 0.55;
}
`

export const LanguageToggle = () => {
  const Component = ({ fileData, allFiles }) => {
    const currentSlug = fileData.slug
    if (!currentSlug || currentSlug === "404") return null

    const englishPage = isEnglishSlug(currentSlug)
    const counterpart = translatedSlug(currentSlug)
    const hasCounterpart = allFiles.some((file) => file.slug === counterpart)
    const targetHref = hasCounterpart ? resolveRelative(currentSlug, counterpart) : undefined

    const chineseControl = englishPage
      ? hasCounterpart
        ? control("a", "中文", { href: targetHref, lang: "zh-CN" })
        : control("span", "中文", {
            class: "is-disabled",
            lang: "zh-CN",
            "aria-disabled": "true",
          })
      : control("span", "中文", {
          class: "is-active",
          lang: "zh-CN",
          "aria-current": "page",
        })

    const englishControl = englishPage
      ? control("span", "EN", {
          class: "is-active",
          lang: "en",
          "aria-current": "page",
        })
      : hasCounterpart
        ? control("a", "EN", { href: targetHref, lang: "en" })
        : control("span", "EN", {
            class: "is-disabled",
            lang: "en",
            "aria-disabled": "true",
          })

    return h(
      "nav",
      {
        class: "language-toggle",
        "aria-label": englishPage ? "Language" : "语言",
      },
      chineseControl,
      englishControl,
    )
  }

  Component.css = styles
  return Component
}
