import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Compose Tailwind class strings deduplicating conflicts (e.g.
 * `cn("p-2", isLarge && "p-4")` → `"p-4"`). The shadcn convention.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
