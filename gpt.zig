//
// GPT-2 Implementation in Zig
//
// This file implements a GPT-2 model in Zig, based on Karpathy's llm.c project:
// https://github.com/karpathy/llm.c/blob/master/train_gpt2.c
//
// The goal is to create a functional GPT-2 model that can be trained and generate text.
//

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

const VectorSize = 8;
const VecT = @Vector(VectorSize, f32);

fn encoder_forward(out: []f32, inp: []const u32, wte: []const f32, wpe: []const f32, B: u32, T: u32, C: u32, V: u32) void {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    std.debug.assert(out.len == B * T * C);
    std.debug.assert(inp.len == B * T);
    std.debug.assert(wte.len >= V * C);
    std.debug.assert(wpe.len >= T * C);

    for (0..B) |b| {
        for (0..T) |t| {
            // seek to the output position in out[b,t,:]
            const out_bt = out[b * T * C + t * C ..];
            // get the index of the token at inp[b, t]
            const ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            const wte_ix = wte[ix * C ..];
            // seek to the position in wpe corresponding to the position
            const wpe_t = wpe[t * C ..];
            // add the two vectors and store the result in out[b,t,:]
            for (0..C) |i| {
                const val = wte_ix[i] + wpe_t[i];
                std.debug.assert(math.isNormal(val));
                out_bt[i] = val;
            }
        }
    }
}

fn encoder_backward(dwte: []f32, dwpe: []f32, dout: []const f32, inp: []const u32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * C + t * C ..];
            const ix = inp[b * T + t];
            const dwte_ix = dwte[ix * C ..];
            const dwpe_t = dwpe[t * C ..];
            for (0..C) |i| {
                const d = dout_bt[i];
                std.debug.assert(math.isNormal(d));
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

fn layernorm_forward(out: []f32, mean: []f32, rstd: []f32, inp: []const f32, weight: []const f32, bias: []const f32, B: u32, T: u32, C: u32) void {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    std.debug.assert(out.len == B * T * C);
    std.debug.assert(mean.len == B * T);
    std.debug.assert(rstd.len == B * T);
    std.debug.assert(inp.len == B * T * C);
    std.debug.assert(weight.len == C);
    std.debug.assert(bias.len == C);

    const eps = 1e-5;
    for (0..B) |b| {
        for (0..T) |t| {
            // seek to the input position inp[b,t,:]
            const x = inp[b * T * C + t * C ..];
            // calculate the mean
            var m: f32 = 0;
            for (0..C) |i| {
                const val = x[i];
                std.debug.assert(math.isNormal(val));
                m += val;
            }
            m = m / @as(f32, @floatFromInt(C));
            // calculate the variance (without any bias correction)
            var v: f32 = 0;
            for (0..C) |i| {
                const xshift = x[i] - m;
                const xshift_squared = xshift * xshift;
                std.debug.assert(math.isNormal(xshift_squared));
                v += xshift_squared;
            }
            v = v / @as(f32, @floatFromInt(C));
            // calculate the rstd (reciprocal standard deviation)
            const s = 1.0 / math.sqrt(v + eps);
            // seek to the output position in out[b,t,:]
            const out_bt = out[b * T * C + t * C ..];
            for (0..C) |i| {
                const n = (s * (x[i] - m)); // normalize
                const o = n * weight[i] + bias[i]; // scale and shift
                std.debug.assert(math.isNormal(o));
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            std.debug.assert(math.isNormal(m));
            std.debug.assert(math.isNormal(s));
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

fn layernorm_backward(dinp: []f32, dweight: []f32, dbias: []f32, dout: []const f32, inp: []const f32, weight: []const f32, mean: []const f32, rstd: []const f32, B: u32, T: u32, C: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * C + t * C ..];
            const inp_bt = inp[b * T * C + t * C ..];
            const dinp_bt = dinp[b * T * C + t * C ..];
            const mean_bt = mean[b * T + t];
            const rstd_bt = rstd[b * T + t];

            // first: two reduce operations
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

            // now iterate again and accumulate all the gradients
            for (0..C) |i| {
                const norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                const dout_bti = dout_bt[i];
                const dnorm_i = weight[i] * dout_bti;
                std.debug.assert(math.isNormal(norm_bti));
                std.debug.assert(math.isNormal(dout_bti));
                // gradient contribution to bias
                dbias[i] += dout_bti;
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bti;
                // gradient contribution to input
                var dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                std.debug.assert(math.isNormal(dval));
                dinp_bt[i] += dval;
            }
        }
    }
}

fn matmul_forward(out: []f32, inp: []const f32, weight: []const f32, bias: ?[]const f32, B: u32, T: u32, C: u32, OC: u32) void {
    // Perform matrix multiplication: out = inp * weight + bias
    // out is (B,T,OC), inp is (B,T,C), weight is (C,OC), bias is (OC) if present
    // This function chooses between vectorized and naive implementations based on input size
    std.debug.assert(out.len == B * T * OC);
    std.debug.assert(inp.len == B * T * C);
    std.debug.assert(weight.len == C * OC);

    if (C % VectorSize == 0) {
        matmul_forward_vec(out, inp, weight, bias, B, T, C, OC);
    } else {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
    }
}

fn matmul_forward_naive(out: []f32, inp: []const f32, weight: []const f32, bias: ?[]const f32, B: u32, T: u32, C: u32, OC: u32) void {
    // Naive implementation of matrix multiplication
    // This serves as a fallback for when vectorized implementation can't be used
    for (0..B) |b| {
        for (0..T) |t| {
            const bt = b * T + t;
            for (0..OC) |o| {
                var val: f32 = 0.0;
                for (0..C) |i| {
                    val += inp[bt * C + i] * weight[o * C + i];
                }
                if (bias) |bi| {
                    val += bi[o];
                }
                std.debug.assert(math.isNormal(val));
                out[bt * OC + o] = val;
            }
        }
    }
}

fn matmul_forward_vec(out: []f32, inp: []const f32, weight: []const f32, bias: ?[]const f32, B: u32, T: u32, C: u32, OC: u32) void {
    // Vectorized implementation of matrix multiplication
    // Uses SIMD instructions for improved performance
    for (0..B) |b| {
        for (0..T) |t| {
            var out_bt = out[b * T * OC + t * OC ..];
            const inp_bt = inp[b * T * C + t * C ..];
            for (0..OC) |o| {
                var sum: f32 = 0.0;
                for (0..C / VectorSize) |i| {
                    // Load input and weight vectors
                    const inp_bt_v: VecT = inp_bt[i * VectorSize ..][0..VectorSize].*;
                    const wrow_v: VecT = weight[o * C + i * VectorSize ..][0..VectorSize].*;
                    // Perform vector multiplication and sum
                    const res = @reduce(.Add, inp_bt_v * wrow_v);
                    sum += res;
                }
                std.debug.assert(math.isNormal(sum));
                out_bt[o] = sum;
                if (bias) |bi| {
                    out_bt[o] += bi[o];
                }
            }
        }
    }
}

fn matmul_backward(dinp: []f32, dweight: []f32, dbias: ?[]f32, dout: []const f32, inp: []const f32, weight: []const f32, B: u32, T: u32, C: u32, OC: u32) void {
    if (C % VectorSize == 0) {
        matmul_backward_vec(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    } else {
        matmul_backward_naive(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    }
}

fn matmul_backward_vec(dinp: []f32, dweight: []f32, dbias: ?[]f32, dout: []const f32, inp: []const f32, weight: []const f32, B: u32, T: u32, C: u32, OC: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * OC + t * OC ..];
            var dinp_bt_v: VecT = undefined;
            for (0..OC) |o| {
                for (0..C / VectorSize) |i| {
                    dinp_bt_v = dinp[b * T * C + t * C + i * VectorSize ..][0..VectorSize].*;
                    const v_wrow: VecT = weight[o * C + i * VectorSize ..][0..VectorSize].*;
                    const d_t: VecT = @splat(dout_bt[o]);
                    const zero: VecT = @splat(0);
                    dinp_bt_v += @mulAdd(VecT, v_wrow, d_t, zero);
                    const tmp: [VectorSize]f32 = dinp_bt_v;
                    @memcpy(dinp[b * T * C + t * C + i * VectorSize .. b * T * C + t * C + (i + 1) * VectorSize], &tmp);
                }
            }
        }
    }
    for (0..OC) |o| {
        for (0..B) |b| {
            for (0..T) |t| {
                const dout_bt = dout[b * T * OC + t * OC ..];
                for (0..C / VectorSize) |i| {
                    var dv_wrow: VecT = dweight[o * C + i * VectorSize ..][0..VectorSize].*;

                    const inp_bt_v: VecT = inp[b * T * C + t * C + i * VectorSize ..][0..VectorSize].*;
                    const d_t: VecT = @splat(dout_bt[o]);

                    dv_wrow += inp_bt_v * d_t;
                    var tmp: [VectorSize]f32 = dv_wrow;
                    @memcpy(dweight[o * C + i * VectorSize .. o * C + (i + 1) * VectorSize], &tmp);
                }

                if (dbias) |dbi| {
                    dbi[o] += dout_bt[o];
                }
            }
        }
    }
}
fn matmul_backward_naive(dinp: []f32, dweight: []f32, dbias: ?[]f32, dout: []const f32, inp: []const f32, weight: []const f32, B: u32, T: u32, C: u32, OC: u32) void {
    for (0..B) |b| {
        for (0..T) |t| {
            const dout_bt = dout[b * T * OC + t * OC ..];
            const dinp_bt = dinp[b * T * C + t * C ..];
            for (0..OC) |o| {
                const wrow = weight[o * C ..];
                const d = dout_bt[o];
                std.debug.assert(math.isNormal(d));
                for (0..C) |i| {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    for (0..OC) |o| {
        for (0..B) |b| {
            for (0..T) |t| {
                const dout_bt = dout[b * T * OC + t * OC ..];
                const inp_bt = inp[b * T * C + t * C ..];
                const dwrow = dweight[o * C ..];
                const d = dout_bt[o];
                std.debug.assert(math.isNormal(d));
                if (dbias) |bias| {
                    bias[o] += d;
                }
                for (0..C) |i| {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

fn attention_forward(out: []f32, preatt: []f32, att: []f32, inp: []const f32, B: u32, T: u32, C: u32, NH: u32) void {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
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

                // pass 1: calculate query dot key and maxval
                var maxval = -math.floatMin(f32);
                for (0..t + 1) |t2| {
                    const key_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C ..];
                    var val: f32 = 0.0;
                    for (0..hs) |i| {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    maxval = @max(maxval, val);
                    std.debug.assert(math.isNormal(val));
                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                var expsum: f32 = 0.0;
                for (0..t + 1) |t2| {
                    const expv = math.exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    std.debug.assert(math.isNormal(expv));
                    att_bth[t2] = expv;
                }
                const expsum_inv = if (expsum == 0.0) 0.0 else 1.0 / expsum;
                std.debug.assert(math.isNormal(expsum_inv));

                // pass 3: normalize to get the softmax
                for (0..T) |t2| {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                const out_bth = out[b * T * C + t * C + h * hs ..];
                for (0..hs) |i| {
                    out_bth[i] = 0.0;
                }
                for (0..t + 1) |t2| {
                    const value_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    const att_btht2 = att_bth[t2];
                    for (0..hs) |i| {
                        const val = att_btht2 * value_t2[i];
                        std.debug.assert(math.isNormal(val));
                        out_bth[i] += val;
                    }
                }
            }
        }
    }
}

fn attention_backward(dinp: []f32, dpreatt: []f32, datt: []f32, dout: []const f32, inp: []const f32, att: []const f32, B: u32, T: u32, C: u32, NH: u32) void {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
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

                // backward pass 4, through the value accumulation
                const dout_bth = dout[b * T * C + t * C + h * hs ..];
                for (0..t + 1) |t2| {
                    const value_t2 = inp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    var dvalue_t2 = dinp[b * T * C3 + t2 * C3 + h * hs + C * 2 ..];
                    for (0..hs) |i| {
                        const datt_btht2 = value_t2[i] * dout_bth[i];
                        const dvalue_t2i = att_bth[t2] * dout_bth[i];
                        std.debug.assert(math.isNormal(datt_btht2));
                        std.debug.assert(math.isNormal(dvalue_t2i));
                        datt_bth[t2] += datt_btht2;
                        dvalue_t2[i] += dvalue_t2i;
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax doesn't need the input (preatt) to backward
                for (0..t + 1) |t2| {
                    for (0..t + 1) |t3| {
                        var indicator: f32 = 0.0;
                        if (t3 == t2) {
                            indicator = 1.0;
                        }
                        const local_derivative: f32 = att_bth[t2] * (indicator - att_bth[t3]);
                        const val = local_derivative * datt_bth[t2];
                        // std.debug.print("{d} {d} {d}\n", .{ val, local_derivative, datt_bth[t2] });
                        // std.debug.assert(math.isNormal(val));
                        dpreatt_bth[t3] += val;
                    }
                }

                // backward pass 1, the query @ key matmul
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
fn gelu_forward(out: []f32, inp: []const f32, N: u32) void {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (0..N) |i| {
        const x = inp[i];
        const cdf = 0.5 * (1.0 + math.tanh(GeluScalingFactor * (x + 0.044715 * math.pow(f32, x, 3))));
        const val = x * cdf;
        std.debug.assert(std.math.isNormal(val));
        out[i] = val;
    }
}

fn gelu_backward(dinp: []f32, inp: []const f32, dout: []const f32, N: u32) void {
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
        std.debug.assert(math.isNormal(val));
        dinp[i] += val;
    }
}

fn residual_forward(out: []f32, inp1: []const f32, inp2: []const f32, N: u32) void {
    for (0..N) |i| {
        const val = inp1[i] + inp2[i];
        std.debug.assert(math.isNormal(val));
        out[i] = val;
    }
}

fn residual_backward(dinp1: []f32, dinp2: []f32, dout: []const f32, N: u32) void {
    for (0..N) |i| {
        const dout_i = dout[i];
        std.debug.assert(math.isNormal(dout_i));
        dinp1[i] += dout_i;
        dinp2[i] += dout_i;
    }
}

fn softmax_forward(probs: []f32, logits: []const f32, B: u32, T: u32, V: u32, Vp: u32) void {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size

    std.debug.assert(probs.len == B * T * Vp);
    std.debug.assert(logits.len == B * T * Vp);

    for (0..B) |b| {
        for (0..T) |t| {
            // probs <- softmax(logits)
            const logit_bt = logits[b * T * Vp + t * Vp ..];
            var prob_bt = probs[b * T * Vp + t * Vp ..];

            // maxval is only calculated and subtracted for numerical stability
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
            // note we only loop to V,l eaving the padded dimensions
            const expsum_inv: f32 = if (sum != 0) 1.0 / sum else 0;
            for (0..V) |v| {
                std.debug.assert(math.isNormal(expsum_inv));
                prob_bt[v] *= expsum_inv;
            }
        }
    }
}

fn crossentropy_forward(losses: []f32, probs: []const f32, targets: []const u32, B: u32, T: u32, Vp: u32) void {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits

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

fn crossentropy_softmax_backward(dlogits: []f32, dlosses: []const f32, probs: []const f32, targets: []const u32, B: u32, T: u32, V: u32, Vp: u32) void {
    // backwards through both softmax and crossentropy
    for (0..B) |b| {
        for (0..T) |t| {
            const dlogits_bt = dlogits[b * T * Vp + t * Vp ..];
            const probs_bt = probs[b * T * Vp + t * Vp ..];
            const dloss = dlosses[b * T + t];
            const ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so grads there stay zero
            for (0..V) |i| {
                const p = probs_bt[i];
                const indicator: f32 = if (i == ix) 1 else 0;
                const val = (p - indicator) * dloss;
                std.debug.assert(math.isNormal(val));
                dlogits_bt[i] += val;
            }
        }
    }
}

fn ParameterTensors(comptime V: u32, comptime C: u32, comptime maxT: u32, comptime L: u32) type {
    const SIZE = (V * C + maxT * C + L * C * 2 + L * 3 * C * C + L * 3 * C + L * C * C + L * C + L * C * 2 + L * 4 * C * C + L * 4 * C + L * C * 4 * C + L * C + C * 2);

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
    const ParamSize = ParameterTensorsT.size();
    const ActSize = ActivationTensorsT.size();

    const Size = ParamSize * 4 + ActSize * 2;

    return struct {
        const Self = @This();

        config: GPT2Config,
        memory: []f32, // heap-allocated memory since too large for stack
        // the weights (parameters) of the model, and their sizes
        params: ParameterTensorsT,
        // gradients of the weights
        grads: ParameterTensorsT,
        // buffers for the AdamW optimizer
        m: ParameterTensorsT,
        v: ParameterTensorsT,
        // the activations of the model, and their size
        acts: ActivationTensorsT,
        // gradients of the activations
        grads_acts: ActivationTensorsT,
        // other run state configuration
        batch_size: u32, // the batch size (B) of current forward pass
        seq_len: u32, // the sequence length (T) of current forward pass
        inputs: [B * T]u32, // the input tokens for the current forward pass
        targets: [B * T]u32, // the target tokens for the current forward pass
        mean_loss: f32, // after a forward pass with targets, will be populated with the mean loss

        pub fn init(allocator: std.mem.Allocator) !Self {
            const memory = try allocator.alloc(f32, Size);
            errdefer allocator.free(memory);

            var self = Self{
                .config = config,
                .memory = memory, // Set memory here
                .params = undefined,
                .grads = undefined,
                .m = undefined,
                .v = undefined,
                .acts = undefined,
                .grads_acts = undefined,
                .batch_size = B,
                .seq_len = T,
                .inputs = undefined,
                .targets = undefined,
                .mean_loss = -1.0,
            };

            var cursor: []f32 = self.memory;

            self.params = ParameterTensorsT.init(cursor[0..ParamSize]);
            cursor = cursor[ParamSize..];

            self.grads = ParameterTensorsT.init(cursor[0..ParamSize]);
            cursor = cursor[ParamSize..];

            self.m = ParameterTensorsT.init(cursor[0..ParamSize]);
            cursor = cursor[ParamSize..];

            self.v = ParameterTensorsT.init(cursor[0..ParamSize]);
            cursor = cursor[ParamSize..];

            self.acts = ActivationTensorsT.init(cursor[0..ActSize]);
            cursor = cursor[ActSize..];

            self.grads_acts = ActivationTensorsT.init(cursor[0..ActSize]);
            cursor = cursor[ActSize..];

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
            std.debug.print("num_parameters: {}\n", .{ParamSize});

            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.memory);
        }

        pub fn forward(self: *Self, inputs: []u32, targets: ?[]u32) void {
            std.debug.assert(inputs.len == B * T);
            for (inputs) |inp| std.debug.assert(0 <= inp and inp < V);

            if (targets) |tgts| std.debug.assert(tgts.len == B * T);
            if (targets) |tgts| for (tgts) |tgt| std.debug.assert(0 <= tgt and tgt < V);

            // cache inputs/targets
            @memcpy(&self.inputs, inputs);
            if (targets) |tgts| @memcpy(&self.targets, tgts);

            // forward pass
            encoder_forward(self.acts.encoded, inputs, self.params.wte, self.params.wpe, B, T, C, V);

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
                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
                attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
                matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
                residual_forward(l_residual2, residual, l_attproj, B * T * C);
                layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);

                residual = l_residual3;
            }

            layernorm_forward(self.acts.lnf, self.acts.lnf_mean, self.acts.lnf_rstd, residual, self.params.lnfw, self.params.lnfb, B, T, C);
            matmul_forward(self.acts.logits, self.acts.lnf, self.params.wte, null, B, T, C, Vp);
            softmax_forward(self.acts.probs, self.acts.logits, B, T, V, Vp);

            // also forward the cross-entropy loss function if we have the targets
            if (targets) |target| {
                crossentropy_forward(self.acts.losses, self.acts.probs, target, B, T, Vp);
                var mean_loss: f32 = 0.0;
                for (self.acts.losses) |loss| {
                    mean_loss += loss;
                }
                self.mean_loss = mean_loss / @as(f32, @floatFromInt(B * T));
            } else {
                self.mean_loss = -1.0;
            }
        }

        pub fn zero_grad(self: *Self) void {
            inline for (std.meta.fields(ParameterTensorsT)) |field| {
                @memset(@field(self.grads, field.name), 0);
            }
        }

        pub fn backward(self: *Self) void {
            // check we forwarded previously
            std.debug.assert(self.mean_loss != -1);

            // kick off chain rule by filling in dlosses
            const dloss_mean = 1 / @as(f32, @floatFromInt(B * T));
            for (self.grads_acts.losses[0 .. B * T]) |*loss| {
                loss.* = dloss_mean;
            }

            // backward pass: go in the reverse order of the forward pass
            crossentropy_softmax_backward(self.grads_acts.logits, self.grads_acts.losses, self.acts.probs, self.targets[0..], B, T, V, Vp);
            matmul_backward(self.grads_acts.lnf, self.grads.wte, null, self.grads_acts.logits, self.acts.lnf, self.params.wte, B, T, C, Vp);

            const residual = self.acts.residual3[(L - 1) * B * T * C ..][0 .. B * T * C];
            const dresidual = self.grads_acts.residual3[(L - 1) * B * T * C ..][0 .. B * T * C];
            layernorm_backward(dresidual, self.grads.lnfw, self.grads.lnfb, self.grads_acts.lnf, residual, self.params.lnfw, self.acts.lnf_mean, self.acts.lnf_rstd, B, T, C);

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
                residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
                matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
                gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
                matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
                layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
                residual_backward(dresidual_l, dl_attproj, dl_residual2, B * T * C);
                matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
                attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
                matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
                layernorm_backward(dresidual_l, dl_ln1w, dl_ln1b, dl_ln1, residual_l, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
            }

            encoder_backward(self.grads.wte, self.grads.wpe, self.grads_acts.encoded, self.inputs[0..], B, T, C);
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
                    std.debug.assert(math.isNormal(m[i]));

                    // update the second moment (RMSprop)
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
                    std.debug.assert(math.isNormal(v[i]));

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

test "GPT" {
    // init
    const B = 2;
    const T = 5;

    const config = GPT2Config{
        .max_seq_len = 32,
        .vocab_size = 100,
        .padded_vocab_size = 128,
        .num_layers = 2,
        .num_heads = 2,
        .channels = 16,
    };

    const allocator = std.testing.allocator;

    var gpt = try GPT(config, B, T).init(allocator);
    defer gpt.deinit(allocator);

    try std.testing.expectEqual(32, gpt.config.max_seq_len);
    try std.testing.expectEqual(100, gpt.config.vocab_size);
    try std.testing.expectEqual(128, gpt.config.padded_vocab_size);
    try std.testing.expectEqual(2, gpt.config.num_layers);
    try std.testing.expectEqual(2, gpt.config.num_heads);
    try std.testing.expectEqual(16, gpt.config.channels);

    const epsilon = 1e-6;

    for (gpt.params.wte) |val| {
        try std.testing.expect(@abs(val) < 0.1);
    }
    for (gpt.params.wpe) |val| {
        try std.testing.expect(@abs(val) < 0.1);
    }

    for (gpt.params.qkvb) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, epsilon);
    }
    for (gpt.params.attprojb) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, epsilon);
    }

    for (gpt.params.ln1w) |val| {
        try std.testing.expectApproxEqAbs(1.0, val, epsilon);
    }
    for (gpt.params.ln2w) |val| {
        try std.testing.expectApproxEqAbs(1.0, val, epsilon);
    }

    for (gpt.params.ln1b) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, epsilon);
    }
    for (gpt.params.ln2b) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, epsilon);
    }

    var inputs = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var targets = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    gpt.forward(inputs[0..], targets[0..]);

    try std.testing.expect(gpt.mean_loss > 0);

    // Test backward
    gpt.backward();
    var has_nonzero_grad = false;
    for (gpt.grads.wte) |val| {
        if (val != 0) {
            has_nonzero_grad = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero_grad);

    // Test update
    const initial_param = gpt.params.wte[0];
    gpt.update(0.01, 0.9, 0.999, 1e-8, 0.1, 1);
    try std.testing.expect(gpt.params.wte[0] != initial_param);

    gpt.zero_grad();
    for (gpt.grads.wte) |val| {
        try std.testing.expectApproxEqAbs(0.0, val, epsilon);
    }
}

test "encoder" {
    const B: u32 = 2; // Batch size
    const T: u32 = 3; // Sequence length
    const C: u32 = 4; // Channels
    const V: u32 = 5; // Vocab size

    var out: [B * T * C]f32 = undefined;
    var dout: [B * T * C]f32 = undefined;
    var dwte: [V * C]f32 = undefined;
    var dwpe: [T * C]f32 = undefined;
    const inp = [_]u32{ 0, 2, 1, 3, 4, 2 };
    const wte = [_]f32{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0,
    };
    const wpe = [_]f32{
        0.01, 0.02, 0.03, 0.04,
        0.05, 0.06, 0.07, 0.08,
        0.09, 0.10, 0.11, 0.12,
    };

    encoder_forward(&out, &inp, &wte, &wpe, B, T, C, V);

    const expected_out = [_]f32{
        0.11, 0.22, 0.33, 0.44,
        0.95, 1.06, 1.17, 1.28,
        0.59, 0.70, 0.81, 0.92,
        1.31, 1.42, 1.53, 1.64,
        1.75, 1.86, 1.97, 2.08,
        0.99, 1.10, 1.21, 1.32,
    };

    const epsilon = 0.0001;
    for (out, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_out[i], value, epsilon);
    }

    // set dout to some arbitrary values
    for (0..B * T * C) |i| {
        dout[i] = @floatFromInt(i % 7);
    }

    @memset(&dwte, 0);
    @memset(&dwpe, 0);

    encoder_backward(&dwte, &dwpe, &dout, &inp, B, T, C);

    const expected_dwte = [_]f32{
        0,  1, 2, 3,
        1,  2, 3, 4,
        10, 5, 7, 2,
        5,  6, 0, 1,
        2,  3, 4, 5,
    };

    const expected_dwpe = [_]f32{
        5, 7, 2,  4,
        6, 8, 10, 5,
        7, 2, 4,  6,
    };

    for (dwte, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dwte[i], value, epsilon);
    }
    for (dwpe, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dwpe[i], value, epsilon);
    }
}

test "layernorm" {
    const B: u32 = 2; // Batch size
    const T: u32 = 3; // Sequence length
    const C: u32 = 4; // Channels

    var out: [B * T * C]f32 = undefined;
    var mean: [B * T]f32 = undefined;
    var rstd: [B * T]f32 = undefined;
    var dinp: [B * T * C]f32 = undefined;
    var dweight: [C]f32 = undefined;
    var dbias: [C]f32 = undefined;
    const inp = [_]f32{
        0.1,  -0.2, 0.3,  0.4,
        0.5,  0.6,  -0.7, 0.8,
        -0.9, 1.0,  0.1,  0.2,
        0.3,  -0.4, 0.5,  -0.6,
        0.7,  0.8,  -0.9, 1.0,
        -0.1, 0.2,  -0.3, 0.4,
    };
    const weight = [_]f32{ 1.0, 0.5, 2.0, 1.5 };
    const bias = [_]f32{ 0.1, 0.2, -0.1, 0.3 };

    layernorm_forward(&out, &mean, &rstd, &inp, &weight, &bias, B, T, C);

    const expected_out = [_]f32{
        -0.11819712, -0.56368995, 1.20918262,  1.93647838,
        0.44049761,  0.45537323,  -3.50497627, 1.57686615,
        -1.38248241, 0.86711705,  -0.10000002, 0.52237236,
        0.85923880,  -0.17961936, 2.28617907,  -1.48963428,
        0.49562484,  0.46374989,  -3.52874827, 1.48687458,
        -0.45704761, 0.47852379,  -2.69955540, 2.24966669,
    };
    const expected_mean = [_]f32{
        0.15,  0.30, 0.10,
        -0.05, 0.40, 0.05,
    };
    const expected_rstd = [_]f32{
        4.36394215, 1.70248842, 1.48248243,
        2.16925359, 1.31874943, 3.71365047,
    };

    const epsilon = 0.00001;
    for (out, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_out[i], value, epsilon);
    }
    for (mean, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_mean[i], value, epsilon);
    }
    for (rstd, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_rstd[i], value, epsilon);
    }

    // set dout to some arbitrary values
    var dout: [B * T * C]f32 = undefined;
    for (0..B * T * C) |i| {
        dout[i] = @floatFromInt(i % 5);
    }

    @memset(&dinp, 0);
    @memset(&dweight, 0);
    @memset(&dbias, 0);

    layernorm_backward(&dinp, &dweight, &dbias, &dout, &inp, &weight, &mean, &rstd, B, T, C);

    const expected_dinp = [_]f32{
        -8.2086735, 3.634472,   2.8063116,   1.7678909,
        2.9053354,  -3.9416268, -0.05552673, 1.0918188,
        1.1831539,  1.3256643,  -2.409034,   -0.09978437,
        -5.984396,  1.1035757,  4.6960974,   0.18472338,
        -2.6948466, -2.494171,  0.6880894,   4.5009274,
        -7.875566,  -6.9790354, 7.619232,    7.2353687,
    };
    const expected_dweight = [_]f32{ -1.1713544, 3.1438882, -3.3636255, 12.1880245 };
    const expected_dbias = [_]f32{ 10.0, 11.0, 12.0, 13.0 };

    for (dinp, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dinp[i], value, epsilon);
    }
    for (dweight, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dweight[i], value, epsilon);
    }
    for (dbias, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dbias[i], value, epsilon);
    }
}

test "matmul" {
    const B: u32 = 2; // Batch size
    const T: u32 = 3; // Sequence length
    const C: u32 = 8; // Input channels (multiple of VECTOR_SIZE for vectorized version)
    const OC: u32 = 4; // Output channels

    var out: [B * T * OC]f32 = undefined;
    var dinp: [B * T * C]f32 = undefined;
    var dweight: [C * OC]f32 = undefined;
    var dbias: [OC]f32 = undefined;
    const inp = [_]f32{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
        3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
    };
    const weight = [_]f32{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
    };
    const bias = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    matmul_forward(&out, &inp, &weight, &bias, B, T, C, OC);

    const expected_out = [_]f32{
        2.14,  5.12,  8.10,  11.08,
        5.02,  13.12, 21.22, 29.32,
        7.90,  21.12, 34.34, 47.56,
        10.78, 29.12, 47.46, 65.8,
        13.66, 37.12, 60.58, 84.04,
        16.54, 45.12, 73.70, 102.28,
    };

    const epsilon = 0.0001;
    for (out, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_out[i], value, epsilon);
    }

    // Set dout to some arbitrary values
    var dout: [B * T * OC]f32 = undefined;
    for (0..B * T * OC) |i| {
        dout[i] = @floatFromInt(i % 3);
    }

    @memset(&dinp, 0);
    @memset(&dweight, 0);
    @memset(&dbias, 0);

    matmul_backward(&dinp, &dweight, &dbias, &dout, &inp, &weight, B, T, C, OC);

    const expected_dinp = [_]f32{
        4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4,
        4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2,
        6.9, 7.4, 7.9, 8.4, 8.9, 9.4, 9.9, 10.4,
        4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4,
        4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2,
        6.9, 7.4, 7.9, 8.4, 8.9, 9.4, 9.9, 10.4,
    };
    const expected_dweight = [_]f32{
        15.8, 16.4, 17.0, 17.6, 18.2, 18.8, 19.4, 20.0,
        11.0, 11.6, 12.2, 12.8, 13.4, 14.0, 14.6, 15.2,
        11.0, 11.6, 12.2, 12.8, 13.4, 14.0, 14.6, 15.2,
        15.8, 16.4, 17.0, 17.6, 18.2, 18.8, 19.4, 20.0,
    };
    const expected_dbias = [_]f32{ 6.0, 6.0, 6.0, 6.0 };

    for (dinp, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dinp[i], value, epsilon);
    }
    for (dweight, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dweight[i], value, epsilon);
    }
    for (dbias, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dbias[i], value, epsilon);
    }
}

test "attention" {
    const B: u32 = 2; // Batch size
    const T: u32 = 3; // Sequence length
    const C: u32 = 4; // Channels
    const NH: u32 = 2; // Number of heads

    var out: [B * T * C]f32 = undefined;
    var preatt: [B * NH * T * T]f32 = undefined;
    var att: [B * NH * T * T]f32 = undefined;
    var dinp: [B * T * C * 3]f32 = undefined;
    var dpreatt: [B * NH * T * T]f32 = undefined;
    var datt: [B * NH * T * T]f32 = undefined;
    const inp = [_]f32{
        // Batch 1
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, // Q
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, // K
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, // V
        // Batch 2
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, // Q
        1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, // K
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, // V
    };

    attention_forward(&out, &preatt, &att, &inp, B, T, C, NH);

    const expected_out = [_]f32{
        0.9000, 1.0000, 1.1000, 1.2000,
        0.9509, 1.0509, 1.1516, 1.2516,
        1.0033, 1.1033, 1.2052, 1.3052,
        1.9000, 2.0000, 2.1000, 2.2000,
        1.9544, 2.0544, 2.1551, 2.2551,
        2.0127, 2.1127, 2.2145, 2.3145,
    };

    const expected_preatt = [_]f32{
        0.1202, -math.floatMin(f32), -math.floatMin(f32),
        0.1980, 0.2333,              -math.floatMin(f32),
        0.2758, 0.3253,              0.3748,
        0.3748, -math.floatMin(f32), -math.floatMin(f32),
        0.4808, 0.5445,              -math.floatMin(f32),
        0.5869, 0.6647,              0.7425,
        2.5244, -math.floatMin(f32), -math.floatMin(f32),
        2.7436, 2.9204,              -math.floatMin(f32),
        2.9628, 3.1537,              3.3446,
        3.3446, -math.floatMin(f32), -math.floatMin(f32),
        3.5921, 3.7972,              -math.floatMin(f32),
        3.8396, 4.0588,              4.2780,
    };

    const expected_att = [_]f32{
        1.0000, 0.0000, 0.0000,
        0.4912, 0.5088, 0.0000,
        0.3170, 0.3331, 0.3500,
        1.0000, 0.0000, 0.0000,
        0.4841, 0.5159, 0.0000,
        0.3078, 0.3327, 0.3596,
        1.0000, 0.0000, 0.0000,
        0.4559, 0.5441, 0.0000,
        0.2721, 0.3293, 0.3986,
        1.0000, 0.0000, 0.0000,
        0.4489, 0.5511, 0.0000,
        0.2635, 0.3281, 0.4085,
    };

    const epsilon = 0.0001;
    for (out, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_out[i], value, epsilon);
    }
    for (preatt, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_preatt[i], value, epsilon);
    }
    for (att, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_att[i], value, epsilon);
    }

    const dout = [_]f32{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4,
    };

    @memset(&dinp, 0);
    @memset(&dpreatt, 0);
    @memset(&datt, 0);

    attention_backward(&dinp, &dpreatt, &datt, &dout, &inp, &att, B, T, C, NH);

    // We'll check a few sample values for brevity
    const expected_dinp = [_]f32{
        0.0000,  0.0000,  0.0000,  0.0000,
        -0.0171, -0.0234, -0.0369, -0.0448,
        0.6309,  0.8117,  0.9774,  1.1566,
        0.0019,  0.0019,  0.0026,  0.0026,
        0.0034,  0.0052,  0.0092,  0.0116,
        0.5542,  0.6384,  0.7271,  0.8119,
        0.0089,  0.0089,  0.0108,  0.0108,
        0.0136,  0.0182,  0.0277,  0.0333,
        0.3150,  0.3500,  0.3955,  0.4315,
        0.0000,  0.0000,  0.0000,  0.0000,
        -0.1948, -0.2103, -0.2459, -0.2628,
        2.6464,  2.8192,  2.9589,  3.1302,
        0.0061,  0.0061,  0.0068,  0.0068,
        0.0572,  0.0621,  0.0718,  0.0770,
        1.6165,  1.7038,  1.8016,  1.8895,
        0.0199,  0.0199,  0.0216,  0.0216,
        0.1376,  0.1482,  0.1741,  0.1857,
        0.8371,  0.8769,  0.9395,  0.9803,
    };

    const expected_dpreatt = [_]f32{
        0.0000,  0.0000,  0.0000,
        -0.0275, 0.0275,  0.0000,
        -0.0622, -0.0021, 0.0643,
        0.0000,  0.0000,  0.0000,
        -0.0375, 0.0375,  0.0000,
        -0.0745, -0.0040, 0.0784,
        0.0000,  0.0000,  0.0000,
        -0.0868, 0.0868,  0.0000,
        -0.1318, -0.0179, 0.1497,
        0.0000,  0.0000,  0.0000,
        -0.0965, 0.0965,  0.0000,
        -0.1418, -0.0224, 0.1641,
    };

    const expected_datt = [_]f32{
        0.2900,  0.0000,  0.0000,
        1.0500,  1.1600,  0.0000,
        1.8100,  2.0000,  2.1900,
        0.8100,  0.0000,  0.0000,
        1.7300,  1.8800,  0.0000,
        2.6500,  2.8800,  3.1100,
        5.2700,  0.0000,  0.0000,
        6.8300,  7.1800,  0.0000,
        8.3900,  8.8200,  9.2500,
        6.6700,  0.0000,  0.0000,
        8.3900,  8.7800,  0.0000,
        10.1100, 10.5800, 11.0500,
    };

    for (dinp, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dinp[i], value, epsilon);
    }

    for (dpreatt, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dpreatt[i], value, epsilon);
    }

    for (datt, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_datt[i], value, epsilon);
    }
}

test "residual" {
    const N: u32 = 6;

    var out: [N]f32 = undefined;
    const inp1 = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const inp2 = [_]f32{ 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 };

    residual_forward(&out, &inp1, &inp2, N);

    const expected_out = [_]f32{ 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 };

    const epsilon = 0.0001;
    for (out, expected_out) |value, expected| {
        try std.testing.expectApproxEqAbs(expected, value, epsilon);
    }

    var dinp1: [N]f32 = undefined;
    var dinp2: [N]f32 = undefined;
    const dout = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

    @memset(&dinp1, 0);
    @memset(&dinp2, 0);

    residual_backward(&dinp1, &dinp2, &dout, N);

    const expected_dinp = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

    for (dinp1, dinp2, expected_dinp) |v1, v2, expected| {
        try std.testing.expectApproxEqAbs(expected, v1, epsilon);
        try std.testing.expectApproxEqAbs(expected, v2, epsilon);
    }
}

test "gelu" {
    const N: u32 = 6;

    var out: [N]f32 = undefined;
    const inp = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

    gelu_forward(&out, &inp, N);

    const expected_out = [_]f32{
        0.054, 0.1159, 0.1854, 0.2622, 0.3457, 0.4354,
    };

    const epsilon = 0.0001;
    for (out, expected_out) |value, expected| {
        try std.testing.expectApproxEqAbs(expected, value, epsilon);
    }

    var dinp: [N]f32 = undefined;
    const dout = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

    @memset(&dinp, 0);

    gelu_backward(&dinp, &inp, &dout, N);

    const expected_dinp = [_]f32{ 0.05795, 0.13149, 0.21969, 0.32106, 0.43368, 0.55530 };

    for (dinp, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dinp[i], value, epsilon);
    }
}

test "softmax_forward" {
    const B: u32 = 2; // Batch size
    const T: u32 = 2; // Sequence length
    const V: u32 = 3; // Vocabulary size
    const Vp: u32 = 3; // Padded vocabulary size

    var out: [B * T * V]f32 = undefined;
    const inp = [_]f32{
        // Batch 1
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        // Batch 2
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2,
    };

    softmax_forward(&out, &inp, B, T, V, Vp);

    const expected_out = [_]f32{
        0.3006, 0.3322, 0.3672,
        0.3006, 0.3322, 0.3672,
        0.3006, 0.3322, 0.3672,
        0.3006, 0.3322, 0.3672,
    };

    const epsilon = 0.0001;
    for (out, expected_out) |value, expected| {
        try std.testing.expectApproxEqAbs(expected, value, epsilon);
    }
}

test "softmax_and_crossentropy" {
    const B: u32 = 2; // Batch size
    const T: u32 = 3; // Sequence length
    const V: u32 = 4; // Vocabulary size
    const Vp: u32 = 4; // Padded vocabulary size

    var probs: [B * T * Vp]f32 = undefined;
    const logits = [_]f32{
        1.0, 2.0, 3.0, 4.0,
        2.0, 3.0, 4.0, 1.0,
        3.0, 4.0, 1.0, 2.0,
        4.0, 1.0, 2.0, 3.0,
        1.0, 3.0, 4.0, 2.0,
        2.0, 4.0, 3.0, 1.0,
    };

    softmax_forward(&probs, &logits, B, T, V, Vp);

    const expected_probs = [_]f32{
        0.0321, 0.0871, 0.2369, 0.6439,
        0.0871, 0.2369, 0.6439, 0.0321,
        0.2369, 0.6439, 0.0321, 0.0871,
        0.6439, 0.0321, 0.0871, 0.2369,
        0.0321, 0.2369, 0.6439, 0.0871,
        0.0871, 0.6439, 0.2369, 0.0321,
    };

    const epsilon = 0.0001;
    for (probs, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_probs[i], value, epsilon);
    }

    var losses: [B * T]f32 = undefined;
    const targets = [_]u32{ 2, 1, 3, 0, 2, 1 };

    crossentropy_forward(&losses, &probs, &targets, B, T, Vp);

    const expected_losses = [_]f32{
        1.4402, 1.4402, 2.4402,
        0.4402, 0.4402, 0.4402,
    };

    for (losses, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_losses[i], value, epsilon);
    }

    var dlogits: [B * T * Vp]f32 = undefined;
    const dlosses = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    @memset(&dlogits, 0);

    crossentropy_softmax_backward(&dlogits, &dlosses, &probs, &targets, B, T, V, Vp);

    const expected_dlogits = [_]f32{
        0.0321,  0.0871,  -0.7631, 0.6439,
        0.0871,  -0.7631, 0.6439,  0.0321,
        0.2369,  0.6439,  0.0321,  -0.9129,
        -0.3561, 0.0321,  0.0871,  0.2369,
        0.0321,  0.2369,  -0.3561, 0.0871,
        0.0871,  -0.3561, 0.2369,  0.0321,
    };

    for (dlogits, 0..) |value, i| {
        try std.testing.expectApproxEqAbs(expected_dlogits[i], value, epsilon);
    }
}
