import { Inventory } from '../models/inventory';
import { Prediction } from '../models/prediction';

export function generatePrediction(inventoryData: Inventory[]): Prediction[] {
    const predictions: Prediction[] = [];

    // Example algorithm for generating predictions
    inventoryData.forEach(item => {
        const predictedQuantity = item.quantity * 1.1; // Simple prediction logic
        const prediction = new Prediction(item.id, predictedQuantity, new Date());
        predictions.push(prediction);
    });

    return predictions;
}