/**
 * Deep Life Sci's D-and-helix mark. The mask cuts the helix out of the D so it
 * stays transparent on any surface. useId keeps multiple instances independent.
 * Explicit dimensions serve the header; the square viewBox also supports h-8.
 */
import { useId } from "react";

export function DeepHelixSVG({
  className,
  width,
  height,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  const maskId = useId();
  return (
    <svg
      width={width}
      height={height}
      className={`aspect-square text-[#087f78] dark:text-[#5ee0c6] ${className ?? ""}`}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <mask
          id={maskId}
          x="0"
          y="0"
          width="64"
          height="64"
          maskUnits="userSpaceOnUse"
        >
          <rect width="64" height="64" fill="white" />
          <path
            d="M25 16C44 23 44 41 25 48M39 16C20 23 20 41 39 48"
            stroke="black"
            strokeWidth="4.5"
            strokeLinecap="round"
          />
          <path
            d="M27 22H37M27 42H37"
            stroke="black"
            strokeWidth="3"
            strokeLinecap="round"
          />
        </mask>
      </defs>
      <path
        d="M13 9H29C46 9 55 18 55 32S46 55 29 55H13Z"
        fill="currentColor"
        mask={`url(#${maskId})`}
      />
    </svg>
  );
}
