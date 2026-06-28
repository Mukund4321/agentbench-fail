/**
 * AgentBench-Fail: Scheduled benchmark runner
 *
 * Runs on a schedule, dispatches individual task-processing jobs
 * for any benchmark tasks that haven't been run yet today.
 */
import { schedules } from "@trigger.dev/sdk";
import { processTask } from "./process-task.js";

const MODELS = ["claude", "gpt", "openweight"] as const;
const MODES = ["baseline", "corrected"] as const;
const HORIZONS = ["short", "medium", "long"] as const;

// Task IDs for the full benchmark suite
const SHORT_TASKS = Array.from({ length: 15 }, (_, i) => `short_${String(i + 1).padStart(3, "0")}`);
const MEDIUM_TASKS = Array.from({ length: 15 }, (_, i) => `medium_${String(i + 1).padStart(3, "0")}`);
const LONG_TASKS = Array.from({ length: 15 }, (_, i) => `long_${String(i + 1).padStart(3, "0")}`);
const ALL_TASKS = [...SHORT_TASKS, ...MEDIUM_TASKS, ...LONG_TASKS];

export const checkBenchmarkTask = schedules.task({
  id: "agentbench-check",
  // Run weekly on Monday at 06:00 UTC — benchmark takes hours to complete
  cron: "0 6 * * 1",

  run: async () => {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");

    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) throw new Error("OPENAI_API_KEY is not set");

    console.log(`[AgentBench-Fail] Starting weekly benchmark sweep`);
    console.log(`  Tasks: ${ALL_TASKS.length} | Models: ${MODELS.length} | Modes: ${MODES.length}`);
    console.log(`  Total runs to dispatch: ${ALL_TASKS.length * MODELS.length * MODES.length}`);

    let dispatched = 0;
    let skipped = 0;

    for (const model of MODELS) {
      for (const mode of MODES) {
        for (const taskId of ALL_TASKS) {
          const runKey = `${taskId}-${model}-${mode}`;

          try {
            await processTask.trigger(
              { taskId, model, mode },
              {
                // Idempotency key prevents duplicate processing if scheduler fires twice
                idempotencyKey: `agentbench-${runKey}-${getWeekKey()}`,
              }
            );
            dispatched++;
          } catch (err) {
            console.error(`  [ERROR] Could not dispatch ${runKey}:`, err);
            skipped++;
          }
        }
      }
    }

    console.log(`[AgentBench-Fail] Dispatched ${dispatched} tasks, skipped ${skipped}`);
    return { dispatched, skipped, week: getWeekKey() };
  },
});

function getWeekKey(): string {
  const now = new Date();
  const year = now.getUTCFullYear();
  // ISO week number
  const start = new Date(Date.UTC(year, 0, 1));
  const dayOfYear = Math.ceil((now.getTime() - start.getTime()) / 86400000);
  const week = Math.ceil(dayOfYear / 7);
  return `${year}-W${String(week).padStart(2, "0")}`;
}
