export interface Base {
  run(): void;
  label: string;
}

export class Widget implements Base {
  run(): void {
    helper();
  }
}
