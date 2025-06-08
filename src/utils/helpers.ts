// src/utils/helpers.ts

export function calculateAverage(numbers: number[]): number {
    const total = numbers.reduce((acc, num) => acc + num, 0);
    return total / numbers.length;
}

export function formatDate(date: Date): string {
    return date.toISOString().split('T')[0];
}