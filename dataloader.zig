const std = @import("std");
const builtin = @import("builtin");

pub const DataLoader = @This();

B: u32,
T: u32,
tokens_file: std.fs.File,
file_size: u64,
current_position: u64,
batch: []u32,
inputs: []u32,
targets: []u32,
num_batches: u32,

pub fn init(allocator: std.mem.Allocator, filename: []const u8, B: u32, T: u32) !DataLoader {
    var self = DataLoader{
        .B = B,
        .T = T,
        .tokens_file = try std.fs.cwd().openFile(filename, .{ .mode = .read_only }),
        .file_size = undefined,
        .current_position = 0,
        .batch = try allocator.alloc(u32, B * T + 1),
        .inputs = undefined,
        .targets = undefined,
        .num_batches = undefined,
    };

    self.file_size = try self.tokens_file.getEndPos();
    if (self.file_size < (@as(u64, B) * @as(u64, T) + 1) * @sizeOf(u32)) {
        std.debug.print("Error: file size is too small for the batch size and sequence length\n", .{});
        return error.FileTooSmall;
    }

    self.inputs = self.batch[0 .. B * T];
    self.targets = self.batch[1 .. B * T + 1];
    self.num_batches = @intCast(self.file_size / (@as(u64, B) * @as(u64, T) * @sizeOf(u32)));

    return self;
}

pub fn reset(self: *DataLoader) void {
    self.current_position = 0;
}

pub fn next_batch(self: *DataLoader) !void {
    const B: u32 = self.B;
    const T: u32 = self.T;
    if (self.current_position + (B * T + 1) * @sizeOf(i32) >= self.file_size) {
        self.current_position = 0;
    }
    _ = try self.tokens_file.seekTo(self.current_position);
    for (0..B * T + 1) |i| {
        self.batch[i] = try self.tokens_file.reader().readVarInt(u32, builtin.cpu.arch.endian(), 4);
    }
    self.current_position += @as(u64, B) * @as(u64, T) * @sizeOf(i32);
}

pub fn deinit(self: *DataLoader, allocator: std.mem.Allocator) void {
    self.tokens_file.close();
    allocator.free(self.batch);
}
