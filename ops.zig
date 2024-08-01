const std = @import("std");
const math = std.math;

const c = @cImport({
    @cInclude("cblas.h");
});

pub fn encoder_forward(out: []f32, inp: []const u32, wte: []const f32, wpe: []const f32, B: u32, T: u32, C: u32, V: u32) void {
    std.debug.assert(out.len == B * T * C);
    std.debug.assert(inp.len == B * T);
    std.debug.assert(wte.len >= V * C);
    std.debug.assert(wpe.len >= T * C);

    for (0..B) |b| {
        for (0..T) |t| {
            const out_bt = out[b * T * C + t * C ..];
            const ix = inp[b * T + t];
            const wte_ix = wte[ix * C ..];
            const wpe_t = wpe[t * C ..];
            for (0..C) |i| {
                const val = wte_ix[i] + wpe_t[i];
                out_bt[i] = val;
            }
        }
    }
}

pub fn encoder_backward(dwte: []f32, dwpe: []f32, dout: []const f32, inp: []const u32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * C + t * C ..];
            const ix = inp[b * T + t];
            const dwte_ix = dwte[ix * C ..];
            const dwpe_t = dwpe[t * C ..];
            for (0..C) |i| {
                const d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

pub fn layernorm_forward(out: []f32, mean: []f32, rstd: []f32, inp: []const f32, weight: []const f32, bias: []const f32, B: u32, T: u32, C: u32) void {
    std.debug.assert(out.len == B * T * C);
    std.debug.assert(mean.len == B * T);
    std.debug.assert(rstd.len == B * T);
    std.debug.assert(inp.len == B * T * C);
    std.debug.assert(weight.len == C);
    std.debug.assert(bias.len == C);

    const eps = 1e-5;
    for (0..B) |b| {
        for (0..T) |t| {
            const x = inp[b * T * C + t * C ..];
            var m: f32 = 0;
            for (0..C) |i| {
                const val = x[i];
                m += val;
            }
            m = m / @as(f32, @floatFromInt(C));
            var v: f32 = 0;
            for (0..C) |i| {
                const xshift = x[i] - m;
                const xshift_squared = xshift * xshift;
                v += xshift_squared;
            }
            v = v / @as(f32, @floatFromInt(C));
            const s = 1.0 / math.sqrt(v + eps);
            const out_bt = out[b * T * C + t * C ..];
            for (0..C) |i| {
                const n = (s * (x[i] - m)); // normalize
                const o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

pub fn layernorm_backward(dinp: []f32, dweight: []f32, dbias: []f32, dout: []const f32, inp: []const f32, weight: []const f32, mean: []const f32, rstd: []const f32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * C + t * C ..];
            const inp_bt = inp[b * T * C + t * C ..];
            const dinp_bt = dinp[b * T * C + t * C ..];
            const mean_bt = mean[b * T + t];
            const rstd_bt = rstd[b * T + t];

            var dnorm_mean: f32 = 0;
            var dnorm_norm_mean: f32 = 0;
            for (0..C) |i| {
                const norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= @floatFromInt(C);
            dnorm_norm_mean /= @floatFromInt(C);

            for (0..C) |i| {
                const norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const dout_bti = dout_bt[i];
                const dnorm_i = weight[i] * dout_bti;
                dbias[i] += dout_bti;
                dweight[i] += norm_bti * dout_bti;
                var dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

pub fn matmul_forward(out: []f32, inp: []const f32, weight: []const f32, bias: ?[]const f32, B: u32, T: u32, C: u32, OC: u32) void {
    const M: c_int = @intCast(B * T);
    const N: c_int = @intCast(OC);
    const K: c_int = @intCast(C);

    c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans, M, N, K, 1.0, inp.ptr, @intCast(K), weight.ptr, @intCast(K), 0.0, out.ptr, @intCast(N));

    if (bias) |bi| {
        for (0..@intCast(M)) |m| {
            for (0..@intCast(N)) |n| {
                out[m * @as(usize, @intCast(N)) + n] += bi[n];
            }
        }
    }
}

pub fn matmul_backward(dinp: []f32, dweight: []f32, dbias: ?[]f32, dout: []const f32, inp: []const f32, weight: []const f32, B: u32, T: u32, C: u32, OC: u32) void {
    const M: c_int = @intCast(B * T);
    const N: c_int = @intCast(C);
    const K: c_int = @intCast(OC);

    // Compute dinp
    c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, M, N, K, 1.0, dout.ptr, @intCast(K), weight.ptr, @intCast(N), 0.0, dinp.ptr, @intCast(N));

    // Compute dweight
    c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans, K, N, M, 1.0, dout.ptr, @intCast(K), inp.ptr, @intCast(N), 0.0, dweight.ptr, @intCast(N));

    // Compute dbias
    if (dbias) |dbi| {
        for (0..@intCast(K)) |k| {
            var sum: f32 = 0.0;
            for (0..@intCast(M)) |m| {
                sum += dout[m * @as(usize, @intCast(K)) + k];
            }
            dbi[k] += sum;
        }
    }
}

pub fn attention_forward(out: []f32, preatt: []f32, att: []f32, inp: []const f32, B: u32, T: u32, C: u32, NH: u32) void {
    const C3 = C * 3;
    const hs = C / NH; // head size

    std.debug.assert(out.len == B * T * C);
    std.debug.assert(preatt.len == B * NH * T * T);
    std.debug.assert(att.len == B * NH * T * T);
    std.debug.assert(inp.len == B * T * C3);

    const scale = 1 / @sqrt(@as(f32, @floatFromInt(hs)));

    for (0..B) |b| {
        for (0..T) |t| {
            for (0..NH) |h| {
                const query_t = inp[b * T * C3 + t * C3 + h * hs ..];
                const preatt_bth = preatt[b * NH * T * T + h * T * T + t * T ..];
                const att_bth = att[b * NH * T * T + h * T * T + t * T ..];

                var maxval = -math.floatMin(f32);
                for (0..t + 1) |t2| {
                    const key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C ..];
                    var val: f32 = 0.0;
                    for (0..hs) |i| {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    maxval = @max(maxval, val);
                    preatt_bth[t2] = val;
                }

                var expsum: f32 = 0.0;
                for (0..t + 1) |t2| {
                    const expv = math.exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                const expsum_inv = if (expsum == 0.0) 0.0 else 1.0 / expsum;

                for (0..T) |t2| {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0;
                    }
                }

                const out_bth = out[b * T * C + t * C + h * hs ..];
                for (0..hs) |i| {
                    out_bth[i] = 0.0;
                }
                for (0..t + 1) |t2| {
                    const value_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    const att_btht2 = att_bth[t2];
                    for (0..hs) |i| {
                        const val = att_btht2 * value_t2[i];
                        out_bth[i] += val;
                    }
                }
            }
        }
    }
}

pub fn attention_backward(dinp: []f32, dpreatt: []f32, datt: []f32, dout: []const f32, inp: []const f32, att: []const f32, B: u32, T: u32, C: u32, NH: u32) void {
    const C3 = C * 3;
    const hs: u32 = @intCast(C / NH);
    const hs_float: f32 = @floatFromInt(hs);
    const scale: f32 = 1.0 / math.sqrt(hs_float);

    for (0..B) |b| {
        for (0..T) |t| {
            for (0..NH) |h| {
                const att_bth = att[b * NH * T * T + h * T * T + t * T ..];
                var datt_bth = datt[b * NH * T * T + h * T * T + t * T ..];
                var dpreatt_bth = dpreatt[b * NH * T * T + h * T * T + t * T ..];
                var dquery_t = dinp[b * T * C3 + t * C3 + h * hs ..];
                const query_t = inp[b * T * C3 + t * C3 + h * hs ..];

                const dout_bth = dout[b * T * C + t * C + h * hs ..];
                for (0..t + 1) |t2| {
                    const value_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    var dvalue_t2 = dinp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    for (0..hs) |i| {
                        const datt_btht2 = value_t2[i] * dout_bth[i];
                        const dvalue_t2i = att_bth[t2] * dout_bth[i];
                        datt_bth[t2] += datt_btht2;
                        dvalue_t2[i] += dvalue_t2i;
                    }
                }

                for (0..t + 1) |t2| {
                    for (0..t + 1) |t3| {
                        var indicator: f32 = 0.0;
                        if (t3 == t2) {
                            indicator = 1.0;
                        }
                        const local_derivative: f32 = att_bth[t2] * (indicator - att_bth[t3]);
                        const val = local_derivative * datt_bth[t2];
                        dpreatt_bth[t3] += val;
                    }
                }

                for (0..t + 1) |t2| {
                    const key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C ..];
                    var dkey_t2 = dinp[b * T * C3 + t2 * C3 + h * hs + C ..];
                    for (0..hs) |i| {
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

const GeluScalingFactor: f32 = math.sqrt(2.0 / math.pi);
pub fn gelu_forward(out: []f32, inp: []const f32, N: u32) void {
    for (0..N) |i| {
        const x = inp[i];
        const cdf = 0.5 * (1.0 + math.tanh(GeluScalingFactor * (x + 0.044715 * math.pow(f32, x, 3))));
        const val = x * cdf;
        out[i] = val;
    }
}

pub fn gelu_backward(dinp: []f32, inp: []const f32, dout: []const f32, N: u32) void {
    for (0..N) |i| {
        const x = inp[i];
        const square = x * x * 0.044715;
        const cube = square * x;
        const tanh_arg = GeluScalingFactor * (x + cube);
        const tanh_out = math.tanh(tanh_arg);
        const coshf_out = math.cosh(tanh_arg);
        const sech2 = 1 / (coshf_out * coshf_out);
        const local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech2 * GeluScalingFactor * (1.0 + 3.0 * square);
        const val = local_grad * dout[i];
        dinp[i] += val;
    }
}

pub fn residual_forward(out: []f32, inp1: []const f32, inp2: []const f32, N: u32) void {
    for (0..N) |i| {
        const val = inp1[i] + inp2[i];
        out[i] = val;
    }
}

pub fn residual_backward(dinp1: []f32, dinp2: []f32, dout: []const f32, N: u32) void {
    for (0..N) |i| {
        const dout_i = dout[i];
        dinp1[i] += dout_i;
        dinp2[i] += dout_i;
    }
}

pub fn softmax_forward(probs: []f32, logits: []const f32, B: u32, T: u32, V: u32, Vp: u32) void {
    std.debug.assert(probs.len == B * T * Vp);
    std.debug.assert(logits.len == B * T * Vp);

    for (0..B) |b| {
        for (0..T) |t| {
            const logit_bt = logits[b * T * Vp + t * Vp ..];
            var prob_bt = probs[b * T * Vp + t * Vp ..];

            var maxval = -math.floatMax(f32);
            for (0..V) |v| {
                if (logit_bt[v] > maxval) {
                    maxval = logit_bt[v];
                }
            }
            var sum: f32 = 0;
            for (0..V) |v| {
                const expv = math.exp(logit_bt[v] - maxval);
                sum += expv;
                std.debug.assert(math.isNormal(expv));
                prob_bt[v] = expv;
            }
            const expsum_inv: f32 = if (sum != 0) 1.0 / sum else 0;
            for (0..V) |v| {
                std.debug.assert(math.isNormal(expsum_inv));
                prob_bt[v] *= expsum_inv;
            }
        }
    }
}

pub fn crossentropy_forward(losses: []f32, probs: []const f32, targets: []const u32, B: u32, T: u32, Vp: u32) void {
    std.debug.assert(losses.len == B * T);
    std.debug.assert(probs.len == B * T * Vp);
    std.debug.assert(targets.len == B * T);

    const epsilon: f32 = 1e-10; // small constant to avoid ln(0)
    for (0..B) |b| {
        for (0..T) |t| {
            const probs_bt = probs[b * T * Vp + t * Vp ..];
            const ix = targets[b * T + t];
            const val = -@log(@max(probs_bt[ix], epsilon));
            std.debug.assert(math.isNormal(val));
            losses[b * T + t] = val;
        }
    }
}

pub fn crossentropy_softmax_backward(dlogits: []f32, dlosses: []const f32, probs: []const f32, targets: []const u32, B: u32, T: u32, V: u32, Vp: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dlogits_bt = dlogits[b * T * Vp + t * Vp ..];
            const probs_bt = probs[b * T * Vp + t * Vp ..];
            const dloss = dlosses[b * T + t];
            const ix = targets[b * T + t];
            for (0..V) |i| {
                const p = probs_bt[i];
                const indicator: f32 = if (i == ix) 1 else 0;
                const val = (p - indicator) * dloss;
                dlogits_bt[i] += val;
            }
        }
    }
}

fn is_finite(slice: []const f32) !void {
    for (slice) |value| {
        try std.testing.expect(math.isFinite(value));
    }
}

test "Simple sanity check for all functions" {
    var allocator = std.testing.allocator;
    // Common parameters
    const B: u32 = 2;
    const T: u32 = 3;
    const C: u32 = 4;
    const V: u32 = 5;
    // const Vp: u32 = 8;
    const NH: u32 = 2;
    const OC: u32 = 6;

    // Test encoder_forward
    {
        const out = try allocator.alloc(f32, B * T * C);
        defer allocator.free(out);
        const inp = try allocator.alloc(u32, B * T);
        defer allocator.free(inp);
        const wte = try allocator.alloc(f32, V * C);
        defer allocator.free(wte);
        const wpe = try allocator.alloc(f32, T * C);
        defer allocator.free(wpe);
        @memset(inp, 0);
        @memset(wte, 0.1);
        @memset(wpe, 0.01);
        encoder_forward(out, inp, wte, wpe, B, T, C, V);
        try is_finite(out);
    }

    // Test encoder_backward
    {
        const dwte = try allocator.alloc(f32, V * C);
        defer allocator.free(dwte);
        const dwpe = try allocator.alloc(f32, T * C);
        defer allocator.free(dwpe);
        const dout = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dout);
        const inp = try allocator.alloc(u32, B * T);
        defer allocator.free(inp);
        @memset(dout, 0.1);
        @memset(inp, 0);
        encoder_backward(dwte, dwpe, dout, inp, B, T, C);
        try is_finite(dwte);
        try is_finite(dwpe);
    }

    // Test layernorm_forward
    {
        const out = try allocator.alloc(f32, B * T * C);
        defer allocator.free(out);
        const mean = try allocator.alloc(f32, B * T);
        defer allocator.free(mean);
        const rstd = try allocator.alloc(f32, B * T);
        defer allocator.free(rstd);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        const weight = try allocator.alloc(f32, C);
        defer allocator.free(weight);
        const bias = try allocator.alloc(f32, C);
        defer allocator.free(bias);
        @memset(inp, 0.1);
        @memset(weight, 1.0);
        @memset(bias, 0.0);
        layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);
        try is_finite(out);
        try is_finite(mean);
        try is_finite(rstd);
    }

    // Test layernorm_backward
    {
        const dinp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dinp);
        const dweight = try allocator.alloc(f32, C);
        defer allocator.free(dweight);
        const dbias = try allocator.alloc(f32, C);
        defer allocator.free(dbias);
        const dout = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dout);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        const weight = try allocator.alloc(f32, C);
        defer allocator.free(weight);
        const mean = try allocator.alloc(f32, B * T);
        defer allocator.free(mean);
        const rstd = try allocator.alloc(f32, B * T);
        defer allocator.free(rstd);
        @memset(dout, 0.1);
        @memset(inp, 0.1);
        @memset(weight, 1.0);
        @memset(mean, 0.0);
        @memset(rstd, 1.0);
        layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
        try is_finite(dinp);
        try is_finite(dweight);
        try is_finite(dbias);
    }

    // Test matmul_forward
    {
        const out = try allocator.alloc(f32, B * T * OC);
        defer allocator.free(out);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        const weight = try allocator.alloc(f32, OC * C);
        defer allocator.free(weight);
        const bias = try allocator.alloc(f32, OC);
        defer allocator.free(bias);
        @memset(inp, 0.1);
        @memset(weight, 0.1);
        @memset(bias, 0.1);
        matmul_forward(out, inp, weight, bias, B, T, C, OC);
        try is_finite(out);
    }

    // Test matmul_backward
    {
        const dinp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dinp);
        const dweight = try allocator.alloc(f32, OC * C);
        defer allocator.free(dweight);
        const dbias = try allocator.alloc(f32, OC);
        defer allocator.free(dbias);
        const dout = try allocator.alloc(f32, B * T * OC);
        defer allocator.free(dout);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        const weight = try allocator.alloc(f32, OC * C);
        defer allocator.free(weight);
        @memset(dout, 0.1);
        @memset(inp, 0.1);
        @memset(weight, 0.1);
        matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
        try is_finite(dinp);
        try is_finite(dweight);
        try is_finite(dbias);
    }

    // Test attention_forward
    {
        const out = try allocator.alloc(f32, B * T * C);
        defer allocator.free(out);
        const preatt = try allocator.alloc(f32, B * NH * T * T);
        defer allocator.free(preatt);
        const att = try allocator.alloc(f32, B * NH * T * T);
        defer allocator.free(att);
        const inp = try allocator.alloc(f32, B * T * C * 3);
        defer allocator.free(inp);
        @memset(inp, 0.1);
        attention_forward(out, preatt, att, inp, B, T, C, NH);
        try is_finite(out);
        try is_finite(preatt);
        try is_finite(att);
    }

    // Test attention_backward
    {
        const dinp = try allocator.alloc(f32, B * T * C * 3);
        defer allocator.free(dinp);
        const dpreatt = try allocator.alloc(f32, B * NH * T * T);
        defer allocator.free(dpreatt);
        const datt = try allocator.alloc(f32, B * NH * T * T);
        defer allocator.free(datt);
        const dout = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dout);
        const inp = try allocator.alloc(f32, B * T * C * 3);
        defer allocator.free(inp);
        const att = try allocator.alloc(f32, B * NH * T * T);
        defer allocator.free(att);
        @memset(dout, 0.1);
        @memset(inp, 0.1);
        @memset(att, 0.1);
        attention_backward(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
        try is_finite(dinp);
        try is_finite(dpreatt);
        try is_finite(datt);
    }

    // Test gelu_forward
    {
        const out = try allocator.alloc(f32, B * T * C);
        defer allocator.free(out);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        @memset(inp, 0.1);
        gelu_forward(out, inp, B * T * C);
        try is_finite(out);
    }

    // Test gelu_backward
    {
        const dinp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dinp);
        const inp = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp);
        const dout = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dout);
        @memset(inp, 0.1);
        @memset(dout, 0.1);
        gelu_backward(dinp, inp, dout, B * T * C);
        try is_finite(dinp);
    }

    // Test residual_forward
    {
        const out = try allocator.alloc(f32, B * T * C);
        defer allocator.free(out);
        const inp1 = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp1);
        const inp2 = try allocator.alloc(f32, B * T * C);
        defer allocator.free(inp2);
        @memset(inp1, 0.1);
        @memset(inp2, 0.1);
        residual_forward(out, inp1, inp2, B * T * C);
        try is_finite(out);
    }

    // Test residual_backward
    {
        const dinp1 = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dinp1);
        const dinp2 = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dinp2);
        const dout = try allocator.alloc(f32, B * T * C);
        defer allocator.free(dout);
        @memset(dout, 0.1);
        residual_backward(dinp1, dinp2, dout, B * T * C);
        try is_finite(dinp1);
        try is_finite(dinp2);
    }

    // Test softmax_forward
    {
        const probs = try allocator.alloc(f32, B * T * V);
        defer allocator.free(probs);
        const logits = try allocator.alloc(f32, B * T * V);
        defer allocator.free(logits);
        @memset(logits, 0.1);
        softmax_forward(probs, logits, B, T, V, V);
        try is_finite(probs);
    }

    // Test crossentropy_forward
    {
        const losses = try allocator.alloc(f32, B * T);
        defer allocator.free(losses);
        const probs = try allocator.alloc(f32, B * T * V);
        defer allocator.free(probs);
        const targets = try allocator.alloc(u32, B * T);
        defer allocator.free(targets);
        @memset(probs, 0.1);
        @memset(targets, 0);
        crossentropy_forward(losses, probs, targets, B, T, V);
        try is_finite(losses);
    }

    // Test crossentropy_softmax_backward
    {
        const dlogits = try allocator.alloc(f32, B * T * V);
        defer allocator.free(dlogits);
        const dlosses = try allocator.alloc(f32, B * T);
        defer allocator.free(dlosses);
        const probs = try allocator.alloc(f32, B * T * V);
        defer allocator.free(probs);
        const targets = try allocator.alloc(u32, B * T);
        defer allocator.free(targets);
        @memset(dlosses, 0.1);
        @memset(probs, 0.1);
        @memset(targets, 0);
        crossentropy_softmax_backward(dlogits, dlosses, probs, targets, B, T, V, V);
        try is_finite(dlogits);
    }
}
