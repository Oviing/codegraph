import { Widget } from "./base";
import { helper } from "./helper";

export function start(): void {
  const w = new Widget();
  w.run();
  helper();
}
