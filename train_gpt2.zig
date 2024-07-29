const std = @import("std");
const gpt = @import("gpt.zig");
const GPT = gpt.GPT;
const GPT2Config = gpt.GPT2Config;
const DataLoader = @import("dataloader.zig").DataLoader;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

fn sample_mult(probabilities: []f32, n: u32, coin: f32) u32 {
    var cdf: f32 = 0.0;
    for (probabilities[0..n], 0..) |prob, i| {
        cdf += prob;
        if (coin < cdf) {
            return @intCast(i);
        }
    }
    return @intCast(n - 1);
}

pub fn main() !void {
    const B: u32 = 4;
    const T: u32 = 64;

    var allocator = std.heap.page_allocator;

    // Initialize GPT model
    const config = GPT2Config{
        .max_seq_len = 1024,
        .vocab_size = 50257,
        .padded_vocab_size = 50304,
        .num_layers = 12,
        .num_heads = 12,
        .channels = 768,
    };
    var model = try GPT(config, B, T).init(allocator);
    defer model.deinit(allocator);

    // Initialize Tokenizer
    var tokenizer = try Tokenizer.init(allocator, "data/gpt2_tokenizer.bin");
    defer tokenizer.deinit(allocator);

    // Initialize DataLoaders
    const train_tokens = "data/tiny_shakespeare_train.bin";
    const val_tokens = "data/tiny_shakespeare_val.bin";
    var train_loader = try DataLoader.init(allocator, train_tokens, B, T);
    defer train_loader.deinit(allocator);
    var val_loader = try DataLoader.init(allocator, val_tokens, B, T);
    defer val_loader.deinit(allocator);

    std.debug.print("Train dataset num_batches: {}\n", .{train_loader.num_batches});
    std.debug.print("Val dataset num_batches: {}\n", .{val_loader.num_batches});

    const val_num_batches: u32 = 5;
    var rng = std.Random.DefaultPrng.init(1337);
    var gen_tokens = try allocator.alloc(u32, B * T);
    defer allocator.free(gen_tokens);

    var timer = try std.time.Timer.start();
    var total_training_time: u64 = 0;

    for (0..41) |step| {
        // Validation
        if (step % 10 == 0) {
            var val_loss: f32 = 0.0;
            val_loader.reset();
            for (0..val_num_batches) |_| {
                try val_loader.next_batch();
                model.forward(val_loader.inputs, val_loader.targets);
                val_loss += model.mean_loss;
            }
            val_loss /= @as(f32, @floatFromInt(val_num_batches));
            std.debug.print("val loss {d:.6}\n", .{ step, val_loss });
        }

        // Generation
        if (step > 0 and step % 20 == 0) { // and step % 20 == 0) {
            std.debug.print("step {d}: Starting generation...\n", .{step});
            @memset(gen_tokens, 50256); // GPT2_EOT token
            std.debug.print("generated text:\n---\n", .{});
            for (1..64) |t| {
                model.forward(gen_tokens, null);
                const probs = model.acts.probs[(t - 1) * model.config.padded_vocab_size ..];
                const coin = rng.random().float(f32);
                const next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                const token_str = tokenizer.decode(next_token);
                std.debug.print("{s}", .{token_str});
            }
            std.debug.print("\n---\n", .{});
        }

        // Training step
        timer.reset();
        try train_loader.next_batch();
        model.forward(train_loader.inputs, train_loader.targets);
        model.zero_grad();
        model.backward();
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        const time_elapsed_ns = timer.read();
        total_training_time += time_elapsed_ns;

        std.debug.print("step {d}: train loss {d:.6} (took {d:.2} ms)\n", .{ step, model.mean_loss, @as(f64, @floatFromInt(time_elapsed_ns)) / 1e6 });

        if (step % 100 == 0 and step > 0) {
            const avg_time_per_step = @as(f64, @floatFromInt(total_training_time)) / @as(f64, @floatFromInt(step)) / 1e6;
            std.debug.print("average time per step: {d:.2} ms\n", .{avg_time_per_step});
        }
    }
}
