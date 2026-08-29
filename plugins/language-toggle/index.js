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

const uiTranslationScript = `
function setSiteText(selector, value) {
  const element = document.querySelector(selector)
  if (element) element.textContent = value
}

function setSiteAttribute(selector, name, value) {
  const element = document.querySelector(selector)
  if (element) element.setAttribute(name, value)
}

function translateSiteInterface() {
  const slug = document.body?.dataset?.slug ?? ""
  const englishPage = slug === "en/index" || slug.startsWith("en/")

  const messages = englishPage
    ? {
        search: "Search",
        searchContent: "Search content...",
        explore: "Explore",
        colorTheme: "Color theme",
        darkMode: "Dark mode",
        lightMode: "Light mode",
        readerMode: "Reader mode",
        home: "Home",
        properties: "Properties",
      }
    : {
        search: "搜索",
        searchContent: "搜索内容...",
        explore: "探索",
        colorTheme: "颜色主题",
        darkMode: "暗色模式",
        lightMode: "亮色模式",
        readerMode: "阅读模式",
        home: "首页",
        properties: "属性",
      }

  setSiteText(".search-button p", messages.search)
  setSiteAttribute(".search-button", "aria-label", messages.search)
  setSiteAttribute(".search-bar", "aria-label", messages.searchContent)
  setSiteAttribute(".search-bar", "placeholder", messages.searchContent)
  setSiteText(".title-button.explorer-toggle h2", messages.explore)
  setSiteAttribute(".mobile-explorer", "aria-label", messages.explore)
  setSiteAttribute(".darkmode", "aria-label", messages.colorTheme)
  setSiteText(".darkmode .dayIcon title", messages.darkMode)
  setSiteText(".darkmode .nightIcon title", messages.lightMode)
  setSiteAttribute(".darkmode .dayIcon", "aria-label", messages.darkMode)
  setSiteAttribute(".darkmode .nightIcon", "aria-label", messages.lightMode)
  setSiteAttribute(".readermode", "aria-label", messages.readerMode)
  setSiteText(".readermode title", messages.readerMode)
  setSiteAttribute(".readermode svg", "aria-label", messages.readerMode)
  setSiteText(".breadcrumb-container .breadcrumb-element:first-child a", messages.home)
  setSiteText(".note-properties-title", messages.properties)

  document.querySelectorAll(".content-meta time[datetime]").forEach((time) => {
    const date = new Date(time.getAttribute("datetime"))
    if (Number.isNaN(date.getTime())) return
    time.textContent = new Intl.DateTimeFormat(englishPage ? "en-US" : "zh-CN", {
      year: "numeric",
      month: englishPage ? "short" : "numeric",
      day: "numeric",
    }).format(date)
  })

  document.querySelectorAll(".content-meta span").forEach((element) => {
    const count = element.textContent?.match(/\\d+/)?.[0]
    if (!count) return
    element.textContent = englishPage ? count + " min read" : count + " 分钟阅读"
  })

  document.querySelectorAll(".page-listing > p").forEach((element) => {
    const count = element.textContent?.match(/\\d+/)?.[0]
    if (!count) return
    element.textContent = englishPage
      ? count + " notes in this folder."
      : "此文件夹下有" + count + "条笔记。"
  })
}

document.addEventListener("nav", translateSiteInterface)
document.addEventListener("render", translateSiteInterface)
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
  Component.afterDOMLoaded = uiTranslationScript
  return Component
}
