// Use ES Module syntax (type: "module" in package.json)
import "dotenv/config";
import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import axios from "axios";
import { pipeline, cos_sim } from "@xenova/transformers";

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose
  .connect(process.env.MONGO_URI || "mongodb://localhost:27017/ai-interviewer")
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.log("❌ MongoDB Error:", err));

// ---------------------------------------------------------
// AI Model Loading
// ---------------------------------------------------------

// 1. Question Generation (via Hugging Face API)
const HF_MODEL_REPO_ID = process.env.HF_MODEL_REPO_ID; // Fallback to public model
const HF_API_URL = "https://api-inference.huggingface.co/models";
const HF_API_KEY = process.env.HF_API_KEY; // 🛑 SET THIS IN .env

if (!HF_API_KEY) {
  console.warn("⚠️ HF_API_KEY is not set. Question generation will fail.");
} else {
  console.log(`✅ Using HF Model: ${HF_MODEL_REPO_ID}`);
}

// 2. Answer Evaluation (Local Model using @xenova/transformers.js)
// We use a "singleton" pattern to ensure we only load the model once.
class EmbeddingPipeline {
  static task = "feature-extraction";
  static model = "Xenova/all-MiniLM-L6-v2"; // A fast, high-quality model
  static instance = null;
  static loading = false;
  static loadError = null;

  static async getInstance() {
    if (this.instance === null && !this.loading) {
      this.loading = true;
      try {
        console.log("⏳ Loading sentence-transformer model for evaluation...");
        this.instance = await pipeline(this.task, this.model);
        console.log("✅ Evaluation model loaded.");
        this.loadError = null;
      } catch (error) {
        console.error("❌ Failed to load evaluation model:", error.message);
        this.loadError = error;
        this.instance = null;
      } finally {
        this.loading = false;
      }
    }
    return this.instance;
  }
}

// ---------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------

// Track recently generated questions to avoid duplicates
const recentQuestions = new Map(); // key: topic-subTopic-difficulty, value: array of questions

async function generateQuestion(topic, subTopic, difficulty) {
  // Add randomness to the prompt to get different questions
  const randomSeed = Math.floor(Math.random() * 10000);
  const variations = ["Create", "Generate", "Provide", "Design", "Formulate"];
  const randomVariation =
    variations[Math.floor(Math.random() * variations.length)];

  // Gemma 2B format prompt with more specific instructions
  const prompt = `<start_of_turn>user
${randomVariation} a unique ${difficulty} level interview question about ${subTopic} in ${topic}. Make it different from common questions. Return your response in valid JSON format with exactly two fields: "question" and "ideal_answer". Question ID: ${randomSeed}<end_of_turn>
<start_of_turn>model
`;

  try {
    const response = await axios.post(
      `${HF_API_URL}/${HF_MODEL_REPO_ID}`,
      {
        inputs: prompt,
        parameters: {
          max_new_tokens: 300,
          return_full_text: false,
          temperature: 0.9, // Increased for more randomness
          top_p: 0.95, // Increased for more diversity
          top_k: 50,
          do_sample: true,
          repetition_penalty: 1.2, // Penalize repetition
        },
        options: {
          wait_for_model: true,
          use_cache: false, // Disable caching
        },
      },
      {
        headers: {
          Authorization: `Bearer ${HF_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: 30000, // 30 second timeout
      }
    );

    console.log("Raw HF Response:", JSON.stringify(response.data, null, 2));

    // Handle response format
    let textData = "";
    if (Array.isArray(response.data)) {
      textData = response.data[0]?.generated_text || "";
    } else if (response.data.generated_text) {
      textData = response.data.generated_text;
    } else {
      throw new Error("Unexpected response format from Hugging Face");
    }

    // Try to extract JSON from the response
    const match = textData.match(/{[\s\S]*?}/);

    let result;
    if (match) {
      const parsed = JSON.parse(match[0]);
      result = {
        question: parsed.question || "Could not parse question",
        ideal_answer: parsed.ideal_answer || "N/A",
      };
    } else {
      // If no JSON found, create a structured response from the text
      console.warn("No JSON found in response, using raw text");
      result = {
        question:
          textData.trim() ||
          `What are the key concepts of ${subTopic} in ${topic}?`,
        ideal_answer:
          "Please provide a comprehensive answer covering the main concepts.",
      };
    }

    // Store question in history (keep last 5)
    previousQuestions.push(result.question);
    if (previousQuestions.length > 5) {
      previousQuestions.shift();
    }

    return result;
  } catch (error) {
    console.error(
      "AI Generation Error:",
      error.response?.data || error.message
    );

    // Check if it's a model loading error
    if (error.response?.status === 503) {
      return {
        question:
          "The AI model is currently loading. Please try again in a moment.",
        ideal_answer: "N/A",
      };
    }

    // Fallback: Generate a variety of basic questions
    const fallbackQuestions = [
      {
        question: `Explain the concept of ${subTopic} in ${topic} and its practical applications.`,
        ideal_answer: `${subTopic} is an important concept in ${topic}. It involves understanding the core principles and being able to apply them in real-world scenarios.`,
      },
      {
        question: `What are the main advantages and disadvantages of using ${subTopic} in ${topic}?`,
        ideal_answer: `When considering ${subTopic}, it's important to weigh both benefits and drawbacks in the context of ${topic}.`,
      },
      {
        question: `Can you describe a real-world scenario where ${subTopic} would be particularly useful in ${topic}?`,
        ideal_answer: `${subTopic} can be applied in various scenarios within ${topic}, particularly when specific requirements need to be met.`,
      },
      {
        question: `How does ${subTopic} compare to alternative approaches in ${topic}?`,
        ideal_answer: `Understanding the trade-offs between ${subTopic} and other approaches is crucial for making informed decisions in ${topic}.`,
      },
      {
        question: `What are common mistakes developers make when implementing ${subTopic} in ${topic}?`,
        ideal_answer: `Being aware of common pitfalls helps in implementing ${subTopic} correctly in ${topic} projects.`,
      },
    ];

    // Use timestamp to pick a different fallback each time
    const randomIndex =
      Math.floor(Date.now() / 1000) % fallbackQuestions.length;
    return fallbackQuestions[randomIndex];
  }
}

async function evaluateAnswer(userAnswer, idealAnswer) {
  if (!userAnswer || !idealAnswer || idealAnswer === "N/A") {
    return { feedback: "Could not evaluate answer.", score: 0 };
  }

  // Get the singleton instance of the pipeline
  const extractor = await EmbeddingPipeline.getInstance();

  // If model failed to load, use fallback evaluation
  if (!extractor) {
    console.warn("⚠️ Using fallback evaluation (model not available)");
    return fallbackEvaluation(userAnswer, idealAnswer);
  }

  try {
    // Generate embeddings
    const userEmbedding = await extractor(userAnswer, {
      pooling: "mean",
      normalize: true,
    });
    const idealEmbedding = await extractor(idealAnswer, {
      pooling: "mean",
      normalize: true,
    });

    // Calculate Cosine Similarity
    const score = cos_sim(userEmbedding.data, idealEmbedding.data);
    const scorePercent = Math.round(score * 100);

    // Generate feedback
    let feedback = `**Similarity Score: ${scorePercent}%**\n\n`;
    if (score > 0.8) {
      feedback +=
        "Excellent! Your answer is very comprehensive and closely matches the key points.";
    } else if (score > 0.6) {
      feedback +=
        "Good job! You have the right idea, but you might be missing a few key details.";
    } else if (score > 0.4) {
      feedback +=
        "You're on the right track, but your answer is quite different. Try to be more specific.";
    } else {
      feedback +=
        "Your answer seems to be missing the main points. Let's review the ideal answer.";
    }

    return { feedback, score: scorePercent };
  } catch (error) {
    console.error("❌ Error during evaluation:", error.message);
    return fallbackEvaluation(userAnswer, idealAnswer);
  }
}

// Fallback evaluation using simple keyword matching
function fallbackEvaluation(userAnswer, idealAnswer) {
  const userWords = new Set(userAnswer.toLowerCase().match(/\b\w+\b/g) || []);
  const idealWords = idealAnswer.toLowerCase().match(/\b\w+\b/g) || [];

  if (!idealWords || idealWords.length === 0) {
    return { feedback: "Could not evaluate answer.", score: 0 };
  }

  // Calculate word overlap
  const matches = idealWords.filter((word) => userWords.has(word)).length;
  const score = Math.round((matches / idealWords.length) * 100);

  let feedback = `**Estimated Score: ${score}%** (Using fallback evaluation)\n\n`;
  if (score > 70) {
    feedback += "Good! Your answer covers many key concepts.";
  } else if (score > 40) {
    feedback +=
      "You're on the right track, but could expand more on key concepts.";
  } else {
    feedback +=
      "Your answer needs more detail. Review the ideal answer for key points.";
  }

  return { feedback, score };
}

// ---------------------------------------------------------
// Routes
// ---------------------------------------------------------

app.get("/", (req, res) => {
  res.send("AI Interviewer Backend is running!");
});

// 1. Start Interview Route (Generates Question)
app.post("/api/interview/start", async (req, res) => {
  const { topic, subTopic, difficulty } = req.body;

  if (!topic || !subTopic) {
    return res.status(400).json({ error: "Topic and Subtopic are required" });
  }

  console.log(
    `🤖 Generating question for: ${topic} / ${subTopic} (${difficulty})`
  );
  const aiResponse = await generateQuestion(topic, subTopic, difficulty);

  // In a real app, you would save this to a new Interview Session in MongoDB

  res.json(aiResponse);
});

// 2. Evaluate Answer Route (Performs Semantic Evaluation)
app.post("/api/interview/evaluate", async (req, res) => {
  const { userAnswer, idealAnswer } = req.body;

  console.log(`📝 Evaluating answer...`);
  const evaluation = await evaluateAnswer(userAnswer, idealAnswer);

  // Add the ideal answer to the response for the user to see
  evaluation.idealAnswer = idealAnswer;

  res.json(evaluation);
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  // Pre-warm the evaluation model on startup (optional, non-blocking)
  EmbeddingPipeline.getInstance().catch((err) => {
    console.warn("⚠️ Model pre-loading failed, will use fallback evaluation");
  });
});
