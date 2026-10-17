export function helper(): void {}

export class Unrelated {
  go(): void {
    helper();
  }
}
