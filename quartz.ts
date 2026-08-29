import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import { componentRegistry } from "./quartz/components/registry"
import type { ExplorerOptions } from "@quartz-community/explorer"

const languageAwareExplorerFilter: NonNullable<ExplorerOptions["filterFn"]> = (node) => {
  if (node.slugSegment === "tags") return false

  const currentSlug = document.body?.dataset?.slug ?? ""
  const englishPage = currentSlug === "en/index" || currentSlug.startsWith("en/")
  const rootSegment = node.slugSegments?.[0] ?? node.slugSegment

  return englishPage ? rootSegment === "en" : rootSegment !== "en"
}

componentRegistry.setOptionOverrides("@quartz-community/explorer", {
  filterFn: languageAwareExplorerFilter,
})

const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
