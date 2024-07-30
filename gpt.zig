//
// GPT-2 Implementation in Zig
//
// This file implements a GPT-2 model in Zig, based on Karpathy's llm.c project:
// https://github.com/karpathy/llm.c/blob/master/train_gpt2.c
//
// The goal is to create a functional GPT-2 model that can be trained and generate text.
//

const std = @import("std");
const math = std.math;

// const options = @import("options");
const VectorSize: u32 = 8; // options.vector_size;

const ops = @import("ops.zig");

fn ParameterTensors(comptime V: u32, comptime C: u32, comptime maxT: u32, comptime L: u32) type {
    const SIZE =
        V * C + // wte
        maxT * C + // wpe
        L * C + // ln1w
        L * C + // ln1b
        L * 3 * C * C + // qkvw
        L * 3 * C + // qkvb
        L * C * C + // attprojw
        L * C + // attprojb
        L * C + // ln2w
        L * C + // ln2b
        L * 4 * C * C + // fcw
        L * 4 * C + // fcb
        L * C * 4 * C + // fcprojw
        L * C + // fcprojb
        C + // lnfw
        C; // lnfb

    return struct {
        const Self = @This();

        wte: []f32,
        wpe: []f32,
        ln1w: []f32,
        ln1b: []f32,
        qkvw: []f32,
        qkvb: []f32,
        attprojw: []f32,
        attprojb: []f32,
        ln2w: []f32,
        ln2b: []f32,
        fcw: []f32,
        fcb: []f32,
        fcprojw: []f32,
        fcprojb: []f32,
        lnfw: []f32,
        lnfb: []f32,

        pub fn init(memory: []f32) Self {
            var cursor: u32 = 0;
            const self = Self{
                .wte = blk: {
                    const len = V * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .wpe = blk: {
                    const len = maxT * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln1w = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln1b = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .qkvw = blk: {
                    const len = L * 3 * C * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .qkvb = blk: {
                    const len = L * 3 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .attprojw = blk: {
                    const len = L * C * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .attprojb = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln2w = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln2b = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fcw = blk: {
                    const len = L * 4 * C * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fcb = blk: {
                    const len = L * 4 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fcprojw = blk: {
                    const len = L * C * 4 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fcprojb = blk: {
                    const len = L * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .lnfw = blk: {
                    const len = C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .lnfb = blk: {
                    const len = C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
            };

            std.debug.assert(cursor == SIZE);
            return self;
        }

        pub fn size() u32 {
            return SIZE;
        }
    };
}

fn ActivationTensors(comptime B: u32, comptime T: u32, comptime C: u32, comptime L: u32, comptime NH: u32, comptime V: u32) type {
    const SIZE =
        B * T * C + // encoded
        L * B * T * C + // ln1
        L * B * T + // ln1_mean
        L * B * T + // ln1_rstd
        L * B * T * 3 * C + // qkv
        L * B * T * C + // atty
        L * B * NH * T * T + // preatt
        L * B * NH * T * T + // att
        L * B * T * C + // attproj
        L * B * T * C + // residual2
        L * B * T * C + // ln2
        L * B * T + // ln2_mean
        L * B * T + // ln2_rstd
        L * B * T * 4 * C + // fch
        L * B * T * 4 * C + // fch_gelu
        L * B * T * C + // fcproj
        L * B * T * C + // residual3
        B * T * C + // lnf
        B * T + // lnf_mean
        B * T + // lnf_rstd
        B * T * V + // logits
        B * T * V + // probs
        B * T; // losses

    return struct {
        const Self = @This();

        encoded: []f32,
        ln1: []f32,
        ln1_mean: []f32,
        ln1_rstd: []f32,
        qkv: []f32,
        atty: []f32,
        preatt: []f32,
        att: []f32,
        attproj: []f32,
        residual2: []f32,
        ln2: []f32,
        ln2_mean: []f32,
        ln2_rstd: []f32,
        fch: []f32,
        fch_gelu: []f32,
        fcproj: []f32,
        residual3: []f32,
        lnf: []f32,
        lnf_mean: []f32,
        lnf_rstd: []f32,
        logits: []f32,
        probs: []f32,
        losses: []f32,

        pub fn init(memory: []f32) Self {
            var cursor: u32 = 0;
            const self = Self{
                .encoded = blk: {
                    const len = B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln1 = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln1_mean = blk: {
                    const len = L * B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln1_rstd = blk: {
                    const len = L * B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .qkv = blk: {
                    const len = L * B * T * 3 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .atty = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .preatt = blk: {
                    const len = L * B * NH * T * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .att = blk: {
                    const len = L * B * NH * T * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .attproj = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .residual2 = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln2 = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln2_mean = blk: {
                    const len = L * B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .ln2_rstd = blk: {
                    const len = L * B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fch = blk: {
                    const len = L * B * T * 4 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fch_gelu = blk: {
                    const len = L * B * T * 4 * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .fcproj = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .residual3 = blk: {
                    const len = L * B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .lnf = blk: {
                    const len = B * T * C;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .lnf_mean = blk: {
                    const len = B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .lnf_rstd = blk: {
                    const len = B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .logits = blk: {
                    const len = B * T * V;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .probs = blk: {
                    const len = B * T * V;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
                .losses = blk: {
                    const len = B * T;
                    const slice = memory[cursor .. cursor + len];
                    cursor += len;
                    break :blk slice;
                },
            };

            std.debug.assert(cursor == SIZE);
            return self;
        }

        pub fn size() u32 {
            return SIZE;
        }
    };
}

fn init_normal(random: std.Random, arr: []f32, mean: f32, stddev: f32) void {
    for (arr) |*val| {
        val.* = mean + stddev * random.floatNorm(f32);
    }
}

fn init_zeroes(arr: []f32) void {
    @memset(arr, 0);
}

pub const GPT2Config = struct {
    max_seq_len: u32, // max sequence length: e.g. 1024
    vocab_size: u32, // vocab size, e.g. 50257
    padded_vocab_size: u32, // padded to e.g. %128==0, 50304
    num_layers: u32, // number of layers, e.g. 12
    num_heads: u32, // number of heads in attention, e.g. 12
    channels: u32, // number of channels, e.g. 768
};

pub fn GPT(comptime config: GPT2Config, comptime B: u32, comptime T: u32) type {
    const V = config.vocab_size;
    const Vp = config.padded_vocab_size;
    const C = config.channels;
    const maxT = config.max_seq_len;
    const L = config.num_layers;
    const NH = config.num_heads;

    const ParameterTensorsT = ParameterTensors(Vp, C, maxT, L);
    const ActivationTensorsT = ActivationTensors(B, T, C, L, NH, Vp);
    const PARAM_SIZE = ParameterTensorsT.size();
    const ACT_SIZE = ActivationTensorsT.size();

    const SIZE =
        PARAM_SIZE * 4 + // params, grads, m, v
        ACT_SIZE * 2; // acts, grads_acts

    return struct {
        const Self = @This();

        config: GPT2Config,
        memory: []f32, // heap-allocated memory since too large for stack
        params: ParameterTensorsT, // the weights (parameters) of the model, and their sizes
        grads: ParameterTensorsT, // gradients of the weights
        m: ParameterTensorsT, // buffer for the AdamW optimizer
        v: ParameterTensorsT, // buffer for the AdamW optimizer
        acts: ActivationTensorsT, // the activations of the model, and their size
        grads_acts: ActivationTensorsT, // gradients of the activations

        pub fn init(allocator: std.mem.Allocator) !Self {
            const memory = try allocator.alloc(f32, SIZE);
            errdefer allocator.free(memory);

            var self = Self{
                .config = config,
                .memory = memory,
                .params = undefined,
                .grads = undefined,
                .m = undefined,
                .v = undefined,
                .acts = undefined,
                .grads_acts = undefined,
            };

            var cursor: []f32 = self.memory;

            self.params = ParameterTensorsT.init(cursor[0..PARAM_SIZE]);
            cursor = cursor[PARAM_SIZE..];

            self.grads = ParameterTensorsT.init(cursor[0..PARAM_SIZE]);
            cursor = cursor[PARAM_SIZE..];

            self.m = ParameterTensorsT.init(cursor[0..PARAM_SIZE]);
            cursor = cursor[PARAM_SIZE..];

            self.v = ParameterTensorsT.init(cursor[0..PARAM_SIZE]);
            cursor = cursor[PARAM_SIZE..];

            self.acts = ActivationTensorsT.init(cursor[0..ACT_SIZE]);
            cursor = cursor[ACT_SIZE..];

            self.grads_acts = ActivationTensorsT.init(cursor[0..ACT_SIZE]);
            cursor = cursor[ACT_SIZE..];

            std.debug.assert(cursor.len == 0);

            var generator = std.Random.DefaultPrng.init(42);
            const random = std.Random.init(&generator, std.Random.DefaultPrng.fill);

            // initialize parameters with normal distribution
            init_normal(random, self.params.wte, 0, 0.02);
            init_normal(random, self.params.wpe, 0, 0.02);
            init_normal(random, self.params.qkvw, 0, 0.02);
            init_zeroes(self.params.qkvb);
            init_normal(random, self.params.attprojw, 0, 0.02);
            init_zeroes(self.params.attprojb);
            init_normal(random, self.params.fcw, 0, 0.02);
            init_zeroes(self.params.fcb);
            init_normal(random, self.params.fcprojw, 0, 0.02);
            init_zeroes(self.params.fcprojb);

            // initialize layer norms with ones and zeroes
            init_normal(random, self.params.ln1w, 1.0, 0.0);
            init_zeroes(self.params.ln1b);
            init_normal(random, self.params.ln2w, 1.0, 0.0);
            init_zeroes(self.params.ln2b);
            init_normal(random, self.params.lnfw, 1.0, 0.0);
            init_zeroes(self.params.lnfb);

            std.debug.print("GPT-2 Model:\n", .{});
            std.debug.print("max_seq_len: {}\n", .{T});
            std.debug.print("vocab_size: {}\n", .{V});
            std.debug.print("padded_vocab_size: {}\n", .{Vp});
            std.debug.print("num_layers: {}\n", .{L});
            std.debug.print("num_heads: {}\n", .{NH});
            std.debug.print("channels: {}\n", .{C});
            std.debug.print("num_parameters: {}\n", .{PARAM_SIZE});

            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.memory);
        }

        pub fn forward(self: *Self, inputs: []u32, targets: ?[]u32) f32 {
            std.debug.assert(inputs.len == B * T);
            for (inputs) |inp| std.debug.assert(0 <= inp and inp < V);

            if (targets) |tgts| std.debug.assert(tgts.len == B * T);
            if (targets) |tgts| for (tgts) |tgt| std.debug.assert(0 <= tgt and tgt < V);

            // forward pass
            ops.encoder_forward(self.acts.encoded, inputs, self.params.wte, self.params.wpe, B, T, C, V);

            var residual = self.acts.encoded;
            for (0..L) |l| {
                // get weights for this layer
                const l_ln1w = self.params.ln1w[l * C .. (l + 1) * C];
                const l_ln1b = self.params.ln1b[l * C .. (l + 1) * C];
                const l_qkvw = self.params.qkvw[l * 3 * C * C .. (l + 1) * 3 * C * C];
                const l_qkvb = self.params.qkvb[l * 3 * C .. (l + 1) * 3 * C];
                const l_attprojw = self.params.attprojw[l * C * C .. (l + 1) * C * C];
                const l_attprojb = self.params.attprojb[l * C .. (l + 1) * C];
                const l_ln2w = self.params.ln2w[l * C .. (l + 1) * C];
                const l_ln2b = self.params.ln2b[l * C .. (l + 1) * C];
                const l_fcw = self.params.fcw[l * 4 * C * C .. (l + 1) * 4 * C * C];
                const l_fcb = self.params.fcb[l * 4 * C .. (l + 1) * 4 * C];
                const l_fcprojw = self.params.fcprojw[l * C * 4 * C .. (l + 1) * C * 4 * C];
                const l_fcprojb = self.params.fcprojb[l * C .. (l + 1) * C];

                // get activations for this layer
                const l_ln1 = self.acts.ln1[l * B * T * C .. (l + 1) * B * T * C];
                const l_ln1_mean = self.acts.ln1_mean[l * B * T .. (l + 1) * B * T];
                const l_ln1_rstd = self.acts.ln1_rstd[l * B * T .. (l + 1) * B * T];
                const l_qkv = self.acts.qkv[l * B * T * 3 * C .. (l + 1) * B * T * 3 * C];
                const l_atty = self.acts.atty[l * B * T * C .. (l + 1) * B * T * C];
                const l_preatt = self.acts.preatt[l * B * NH * T * T .. (l + 1) * B * NH * T * T];
                const l_att = self.acts.att[l * B * NH * T * T .. (l + 1) * B * NH * T * T];
                const l_attproj = self.acts.attproj[l * B * T * C .. (l + 1) * B * T * C];
                const l_residual2 = self.acts.residual2[l * B * T * C .. (l + 1) * B * T * C];
                const l_ln2 = self.acts.ln2[l * B * T * C .. (l + 1) * B * T * C];
                const l_ln2_mean = self.acts.ln2_mean[l * B * T .. (l + 1) * B * T];
                const l_ln2_rstd = self.acts.ln2_rstd[l * B * T .. (l + 1) * B * T];
                const l_fch = self.acts.fch[l * B * T * 4 * C .. (l + 1) * B * T * 4 * C];
                const l_fch_gelu = self.acts.fch_gelu[l * B * T * 4 * C .. (l + 1) * B * T * 4 * C];
                const l_fcproj = self.acts.fcproj[l * B * T * C .. (l + 1) * B * T * C];
                const l_residual3 = self.acts.residual3[l * B * T * C .. (l + 1) * B * T * C];

                // now do the forward pass
                ops.layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
                if (C % VectorSize == 0 and C > VectorSize) {
                    ops.matmul_forward_vec(VectorSize, l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
                    ops.attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
                    ops.matmul_forward_vec(VectorSize, l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
                    ops.residual_forward(l_residual2, residual, l_attproj, B * T * C);
                    ops.layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
                    ops.matmul_forward_vec(VectorSize, l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                    ops.gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                    ops.matmul_forward_vec(VectorSize, l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                    ops.residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
                } else {
                    ops.matmul_forward_naive(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
                    ops.attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
                    ops.matmul_forward_naive(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
                    ops.residual_forward(l_residual2, residual, l_attproj, B * T * C);
                    ops.layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
                    ops.matmul_forward_naive(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                    ops.gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                    ops.matmul_forward_naive(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                    ops.residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
                }

                residual = l_residual3;
            }

            ops.layernorm_forward(self.acts.lnf, self.acts.lnf_mean, self.acts.lnf_rstd, residual, self.params.lnfw, self.params.lnfb, B, T, C);
            if (C % VectorSize == 0 and C > VectorSize) {
                ops.matmul_forward_vec(VectorSize, self.acts.logits, self.acts.lnf, self.params.wte, null, B, T, C, Vp);
            } else {
                ops.matmul_forward_naive(self.acts.logits, self.acts.lnf, self.params.wte, null, B, T, C, Vp);
            }
            ops.softmax_forward(self.acts.probs, self.acts.logits, B, T, V, Vp);

            // also forward the cross-entropy loss function if we have the targets
            var mean_loss: f32 = 0.0;
            if (targets) |target| {
                ops.crossentropy_forward(self.acts.losses, self.acts.probs, target, B, T, Vp);
                for (self.acts.losses) |loss| {
                    mean_loss += loss;
                }
                mean_loss = mean_loss / @as(f32, @floatFromInt(B * T));
            } else {
                mean_loss = -1.0;
            }

            return mean_loss;
        }

        pub fn zero_grad(self: *Self) void {
            inline for (std.meta.fields(ParameterTensorsT)) |field| {
                @memset(@field(self.grads, field.name), 0);
            }
        }

        pub fn backward(self: *Self, inputs: []u32, targets: []u32) void {
            // kick off chain rule by filling in dlosses
            const dloss_mean = 1 / @as(f32, @floatFromInt(B * T));
            for (self.grads_acts.losses[0 .. B * T]) |*loss| {
                loss.* = dloss_mean;
            }

            // backward pass: go in the reverse order of the forward pass
            ops.crossentropy_softmax_backward(self.grads_acts.logits, self.grads_acts.losses, self.acts.probs, targets[0..], B, T, V, Vp);

            if (C % VectorSize == 0 and C > VectorSize) {
                ops.matmul_backward_vec(VectorSize, self.grads_acts.lnf, self.grads.wte, null, self.grads_acts.logits, self.acts.lnf, self.params.wte, B, T, C, Vp);
            } else {
                ops.matmul_backward_naive(self.grads_acts.lnf, self.grads.wte, null, self.grads_acts.logits, self.acts.lnf, self.params.wte, B, T, C, Vp);
            }

            const residual = self.acts.residual3[(L - 1) * B * T * C ..][0 .. B * T * C];
            const dresidual = self.grads_acts.residual3[(L - 1) * B * T * C ..][0 .. B * T * C];
            ops.layernorm_backward(dresidual, self.grads.lnfw, self.grads.lnfb, self.grads_acts.lnf, residual, self.params.lnfw, self.acts.lnf_mean, self.acts.lnf_rstd, B, T, C);

            for (0..L) |i| {
                const l = L - 1 - i;

                const residual_l = if (l == 0) self.acts.encoded else self.acts.residual3[(l - 1) * B * T * C ..][0 .. B * T * C];
                const dresidual_l = if (l == 0) self.grads_acts.encoded else self.grads_acts.residual3[(l - 1) * B * T * C ..][0 .. B * T * C];

                // get slices for this layer's weights
                const l_ln1w = self.params.ln1w[l * C ..][0..C];
                const l_qkvw = self.params.qkvw[l * 3 * C * C ..][0 .. 3 * C * C];
                const l_attprojw = self.params.attprojw[l * C * C ..][0 .. C * C];
                const l_ln2w = self.params.ln2w[l * C ..][0..C];
                const l_fcw = self.params.fcw[l * 4 * C * C ..][0 .. 4 * C * C];
                const l_fcprojw = self.params.fcprojw[l * C * 4 * C ..][0 .. C * 4 * C];

                // get slices for this layer's gradient weights
                const dl_ln1w = self.grads.ln1w[l * C ..][0..C];
                const dl_ln1b = self.grads.ln1b[l * C ..][0..C];
                const dl_qkvw = self.grads.qkvw[l * 3 * C * C ..][0 .. 3 * C * C];
                const dl_qkvb = self.grads.qkvb[l * 3 * C ..][0 .. 3 * C];
                const dl_attprojw = self.grads.attprojw[l * C * C ..][0 .. C * C];
                const dl_attprojb = self.grads.attprojb[l * C ..][0..C];
                const dl_ln2w = self.grads.ln2w[l * C ..][0..C];
                const dl_ln2b = self.grads.ln2b[l * C ..][0..C];
                const dl_fcw = self.grads.fcw[l * 4 * C * C ..][0 .. 4 * C * C];
                const dl_fcb = self.grads.fcb[l * 4 * C ..][0 .. 4 * C];
                const dl_fcprojw = self.grads.fcprojw[l * C * 4 * C ..][0 .. C * 4 * C];
                const dl_fcprojb = self.grads.fcprojb[l * C ..][0..C];

                // get slices for this layer's activations
                const l_ln1 = self.acts.ln1[l * B * T * C ..][0 .. B * T * C];
                const l_ln1_mean = self.acts.ln1_mean[l * B * T ..][0 .. B * T];
                const l_ln1_rstd = self.acts.ln1_rstd[l * B * T ..][0 .. B * T];
                const l_qkv = self.acts.qkv[l * B * T * 3 * C ..][0 .. B * T * 3 * C];
                const l_atty = self.acts.atty[l * B * T * C ..][0 .. B * T * C];
                const l_att = self.acts.att[l * B * NH * T * T ..][0 .. B * NH * T * T];
                const l_residual2 = self.acts.residual2[l * B * T * C ..][0 .. B * T * C];
                const l_ln2 = self.acts.ln2[l * B * T * C ..][0 .. B * T * C];
                const l_ln2_mean = self.acts.ln2_mean[l * B * T ..][0 .. B * T];
                const l_ln2_rstd = self.acts.ln2_rstd[l * B * T ..][0 .. B * T];
                const l_fch = self.acts.fch[l * B * T * 4 * C ..][0 .. B * T * 4 * C];
                const l_fch_gelu = self.acts.fch_gelu[l * B * T * 4 * C ..][0 .. B * T * 4 * C];

                // get slices for this layer's gradient activations
                const dl_ln1 = self.grads_acts.ln1[l * B * T * C ..][0 .. B * T * C];
                const dl_qkv = self.grads_acts.qkv[l * B * T * 3 * C ..][0 .. B * T * 3 * C];
                const dl_atty = self.grads_acts.atty[l * B * T * C ..][0 .. B * T * C];
                const dl_preatt = self.grads_acts.preatt[l * B * NH * T * T ..][0 .. B * NH * T * T];
                const dl_att = self.grads_acts.att[l * B * NH * T * T ..][0 .. B * NH * T * T];
                const dl_attproj = self.grads_acts.attproj[l * B * T * C ..][0 .. B * T * C];
                const dl_residual2 = self.grads_acts.residual2[l * B * T * C ..][0 .. B * T * C];
                const dl_ln2 = self.grads_acts.ln2[l * B * T * C ..][0 .. B * T * C];
                const dl_fch = self.grads_acts.fch[l * B * T * 4 * C ..][0 .. B * T * 4 * C];
                const dl_fch_gelu = self.grads_acts.fch_gelu[l * B * T * 4 * C ..][0 .. B * T * 4 * C];
                const dl_fcproj = self.grads_acts.fcproj[l * B * T * C ..][0 .. B * T * C];
                const dl_residual3 = self.grads_acts.residual3[l * B * T * C ..][0 .. B * T * C];

                // backprop this layer
                if (C % VectorSize == 0 and C > VectorSize) {
                    ops.residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
                    ops.matmul_backward_vec(VectorSize, dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
                    ops.gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
                    ops.matmul_backward_vec(VectorSize, dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
                    ops.layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
                    ops.residual_backward(dresidual_l, dl_attproj, dl_residual2, B * T * C);
                    ops.matmul_backward_vec(VectorSize, dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
                    ops.attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
                    ops.matmul_backward_vec(VectorSize, dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
                    ops.layernorm_backward(dresidual_l, dl_ln1w, dl_ln1b, dl_ln1, residual_l, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
                } else {
                    ops.residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
                    ops.matmul_backward_naive(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
                    ops.gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
                    ops.matmul_backward_naive(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
                    ops.layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
                    ops.residual_backward(dresidual_l, dl_attproj, dl_residual2, B * T * C);
                    ops.matmul_backward_naive(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
                    ops.attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
                    ops.matmul_backward_naive(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
                    ops.layernorm_backward(dresidual_l, dl_ln1w, dl_ln1b, dl_ln1, residual_l, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
                }
            }

            ops.encoder_backward(self.grads.wte, self.grads.wpe, self.grads_acts.encoded, inputs, B, T, C);
        }

        pub fn update(self: *Self, learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: usize) void {
            // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

            const t_f32: f32 = @floatFromInt(t);

            inline for (std.meta.fields(ParameterTensorsT)) |field| {
                const params = @field(self.params, field.name);
                const grads = @field(self.grads, field.name);
                const m = @field(self.m, field.name);
                const v = @field(self.v, field.name);

                for (params, 0..) |*param, i| {
                    const grad = grads[i];

                    // update the first moment (momentum)
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad;

                    // update the second moment (RMSprop)
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                    // bias-correct both moments
                    const m_hat = m[i] / (1.0 - std.math.pow(f32, beta1, t_f32));
                    const v_hat = v[i] / (1.0 - std.math.pow(f32, beta2, t_f32));

                    // update
                    const denominator = @sqrt(@max(v_hat, 0)) + eps;
                    const step = if (denominator != 0) m_hat / denominator else 0;
                    const val = learning_rate * (step + weight_decay * param.*);
                    param.* -= val;
                }
            }
        }
    };
}
