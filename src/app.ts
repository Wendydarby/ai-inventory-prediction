import express from 'express';
import { fetchInventory } from './services/data';
import { generatePrediction } from './services/predict';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.get('/inventory', async (req, res) => {
    try {
        const inventory = await fetchInventory();
        res.json(inventory);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching inventory' });
    }
});

app.post('/predict', async (req, res) => {
    const inventoryData = req.body;
    try {
        const prediction = await generatePrediction(inventoryData);
        res.json(prediction);
    } catch (error) {
        res.status(500).json({ message: 'Error generating prediction' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});