/**
 * Show LangChain attribution beside the Deep Life Sci product mark. Dimensions
 * describe the LangChain mark. Deep helix is 30% larger to compensate for its
 * internal padding, giving both symbols similar visual weight at either UI size.
 */
import { DeepHelixSVG } from "./deep-helix";
import { LangGraphLogoSVG } from "./langgraph";

export function BrandLogos({
  className,
  width,
  height,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-2 ${className ?? ""}`}
      style={{ height: height ?? width }}
      role="img"
      aria-label="LangChain and Deep Life Sci"
    >
      <span className="inline-flex h-full aspect-square" aria-hidden="true">
        <LangGraphLogoSVG className="h-full w-full" />
      </span>
      <DeepHelixSVG className="h-[130%] w-auto shrink-0" />
    </span>
  );
}
