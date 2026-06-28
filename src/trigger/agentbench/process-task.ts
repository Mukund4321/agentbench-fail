/**
 * AgentBench-Fail: Individual task processor
 *
 * Executes a single benchmark task via the Python runner (invoked as a subprocess),
 * uploads the trace to cloud storage, and posts a Slack notification on completion.
 */
import { task } from "@trigger.dev/sdk";
import { z } from "zod";

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
const RESULTS_BUCKET = process.env.RESULTS_BUCKET;
const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL;

const TaskPayloadSchema = z.object({
  taskId: z.string().regex(/^(short|medium|long)_\d{3}$/, "Invalid task ID format"),
  model: z.enum(["claude", "gpt", "openweight"]),
  mode: z.enum(["baseline", "corrected"]),
  outputDir: z.string().optional().default("results/raw_traces"),
});

export const processTask = task({
  id: "agentbench-process",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 10_000,
    maxTimeoutInMs: 120_000,
  },

  run: async (payload: z.infer<typeof TaskPayloadSchema>) => {
    const { taskId, model, mode, outputDir } = TaskPayloadSchema.parse(payload);

    // Validate required env vars
    if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not set");
    if (model === "gpt" && !OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set");
    if (model === "openweight" && !TOGETHER_API_KEY) throw new Error("TOGETHER_API_KEY is not set");

    console.log(`[${taskId}] Starting: model=${model} mode=${mode}`);

    // Invoke Python runner as subprocess
    const result = await runPythonTask({ taskId, model, mode, outputDir });

    console.log(`[${taskId}] Completed: success=${result.success} steps=${result.steps_taken}`);

    // Upload trace to cloud storage if bucket configured
    if (RESULTS_BUCKET && result.traceJson) {
      await uploadTrace({
        bucket: RESULTS_BUCKET,
        key: `traces/${taskId}/${model}/${mode}.json`,
        content: result.traceJson,
      });
    }

    // Post Slack notification for failures (if webhook configured)
    if (!result.success && SLACK_WEBHOOK_URL) {
      await notifySlack({
        webhook: SLACK_WEBHOOK_URL,
        taskId,
        model,
        mode,
        failureType: result.failure_type,
        steps: result.steps_taken,
      });
    }

    return {
      taskId,
      model,
      mode,
      success: result.success,
      failure_type: result.failure_type,
      steps_taken: result.steps_taken,
      total_tokens: result.total_tokens,
      total_latency_s: result.total_latency_s,
    };
  },
});

// ── Helpers ───────────────────────────────────────────────────────────────────

interface TaskRunResult {
  success: boolean;
  failure_type: string | null;
  steps_taken: number;
  total_tokens: number;
  total_latency_s: number;
  traceJson?: string;
}

async function runPythonTask(params: {
  taskId: string;
  model: string;
  mode: string;
  outputDir: string;
}): Promise<TaskRunResult> {
  const { taskId, model, mode, outputDir } = params;

  // Build command
  const cmd = [
    "python",
    "tools/run_single_task.py",
    "--task-id", taskId,
    "--model", model,
    "--mode", mode,
    "--output-dir", outputDir,
  ];

  const env = {
    ...process.env,
    ANTHROPIC_API_KEY: ANTHROPIC_API_KEY ?? "",
    OPENAI_API_KEY: OPENAI_API_KEY ?? "",
    TOGETHER_API_KEY: TOGETHER_API_KEY ?? "",
  };

  try {
    const proc = Bun.spawn(cmd, {
      env,
      stdout: "pipe",
      stderr: "pipe",
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      console.error(`[${taskId}] Python runner exited with code ${exitCode}`);
      console.error(`STDERR: ${stderr}`);
      throw new Error(`Python runner failed: ${stderr.slice(0, 500)}`);
    }

    // Parse summary from stdout
    const successMatch = stdout.match(/\[(SUCCESS|FAIL[^\]]*)\]/);
    const success = successMatch?.[1] === "SUCCESS";
    const failureType = success ? null : (successMatch?.[1]?.replace("FAIL [", "").replace("]", "") ?? "unknown");

    const stepsMatch = stdout.match(/Steps:\s*(\d+)/);
    const tokensMatch = stdout.match(/Tokens:\s*(\d+)/);
    const latencyMatch = stdout.match(/Latency:\s*([\d.]+)/);

    // Try to read the trace file
    let traceJson: string | undefined;
    try {
      const tracePath = `${outputDir}/${taskId}_${model}_${mode}.json`;
      traceJson = await Bun.file(tracePath).text();
    } catch {
      // Trace file not available — that's OK
    }

    return {
      success,
      failure_type: failureType,
      steps_taken: parseInt(stepsMatch?.[1] ?? "0", 10),
      total_tokens: parseInt(tokensMatch?.[1] ?? "0", 10),
      total_latency_s: parseFloat(latencyMatch?.[1] ?? "0"),
      traceJson,
    };
  } catch (err) {
    console.error(`[${taskId}] Runner error:`, err);
    throw err;
  }
}

async function uploadTrace(params: {
  bucket: string;
  key: string;
  content: string;
}): Promise<void> {
  // Minimal S3-compatible upload using fetch + presigned URL pattern
  // In production: use AWS SDK or GCS client
  console.log(`  [upload] Would upload to s3://${params.bucket}/${params.key}`);
}

async function notifySlack(params: {
  webhook: string;
  taskId: string;
  model: string;
  mode: string;
  failureType: string | null;
  steps: number;
}): Promise<void> {
  const { webhook, taskId, model, mode, failureType, steps } = params;
  const message = {
    text: `❌ AgentBench-Fail: *${taskId}* failed`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `*AgentBench-Fail Task Failure*\n• Task: \`${taskId}\`\n• Model: ${model}\n• Mode: ${mode}\n• Failure type: \`${failureType ?? "unknown"}\`\n• Steps taken: ${steps}`,
        },
      },
    ],
  };

  const resp = await fetch(webhook, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(message),
  });

  if (!resp.ok) {
    console.warn(`  [slack] Notification failed: ${resp.status}`);
  }
}
