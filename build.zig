const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "train_gpt2",
        .root_source_file = b.path("train_gpt2.zig"),
        .target = target,
        .optimize = optimize,
    });

    const thread_count = b.option(u8, "thread_count", "Number of threads (default: 1)") orelse 1;

    const options = b.addOptions();
    options.addOption(u8, "thread_count", thread_count);
    options.addOption(u32, "vector_size", vector_size(b.graph.host));

    exe.root_module.addOptions("options", options);

    b.installArtifact(exe);
}

fn vector_size(target: std.Build.ResolvedTarget) u32 {
    const cpu = target.result.cpu;
    const x86 = std.Target.x86.Feature;
    const aarch64 = std.Target.aarch64.Feature;

    switch (cpu.arch) {
        .x86_64 => {
            if (cpu.features.isEnabled(@intFromEnum(x86.avx512f))) {
                return 16; // 512 bits / 32 bits = 16 f32s
            } else if (cpu.features.isEnabled(@intFromEnum(x86.avx2)) or
                cpu.features.isEnabled(@intFromEnum(x86.avx)))
            {
                return 8; // 256 bits / 32 bits = 8 f32s
            } else if (cpu.features.isEnabled(@intFromEnum(x86.sse))) {
                return 4; // 128 bits / 32 bits = 4 f32s
            } else {
                return 1; // how old is your cpu??
            }
        },
        .aarch64 => {
            if (cpu.features.isEnabled(@intFromEnum(aarch64.neon))) {
                return 4; // 128 bits / 32 bits = 4 f32s
            } else {
                return 1;
            }
        },
        else => return 1,
    }
}
