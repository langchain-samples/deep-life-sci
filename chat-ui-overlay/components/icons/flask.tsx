/**
 * The product mark: an Erlenmeyer flask, drawn to sit beside the app name.
 *
 * Stroked rather than filled, in `currentColor` rather than a literal, so it takes the
 * heading's colour and reads at the same weight as the text it sits next to — which is the
 * whole point of it. It replaced U+1F9EA TEST TUBE, whose full-colour platform-shaded
 * rendering was the one thing on the home screen that looked pasted on.
 *
 * Sized in `em` by the caller (`h-[1em] w-[1em]`), so one component serves the header, the
 * home screen and the setup form at three different type sizes without a size prop.
 */
export function FlaskSVG({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Rim, then the neck and body as one outline, then the fill line. */}
      <path d="M8.2 3h7.6" />
      <path d="M9.5 3v6.6L5.56 18.66A1.6 1.6 0 0 0 6.7 20.4h10.6a1.6 1.6 0 0 0 1.14-2.74L14.5 9.6V3" />
      <path d="M6.7 16.2h10.6" />
    </svg>
  );
}
