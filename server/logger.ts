type Level = "debug" | "info" | "warn" | "error";
const muted = process.env.NODE_ENV === "production";

function log(level: Level, ...args: any[]) {
  // avoid leaking objects with secrets—stringify shallowly if needed
  // eslint-disable-next-line no-console
  (console as any)[level](...args);
}

export const logger = {
  debug: (...args: any[]) => { if (!muted) log("debug", ...args); },
  info:  (...args: any[]) => log("info",  ...args),
  warn:  (...args: any[]) => log("warn",  ...args),
  error: (...args: any[]) => log("error", ...args),
};