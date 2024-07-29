const std = @import("std");

pub const Tokenizer = @This();

vocab_size: u32,
vocab_map: [][]u8,
init_ok: bool,

pub fn init(allocator: std.mem.Allocator, filename: []const u8) !Tokenizer {
    var self = Tokenizer{
        .vocab_size = undefined,
        .vocab_map = undefined,
        .init_ok = false,
    };

    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const header = try file.reader().readInt(u32, .little);
    const version = try file.reader().readInt(u32, .little);
    if (header != 20240328 or version != 2) {
        return error.InvalidTokenizerFile;
    }

    self.vocab_size = try file.reader().readInt(u32, .little);
    self.vocab_map = try allocator.alloc([]u8, self.vocab_size);

    for (0..self.vocab_size) |i| {
        const token_length = try file.reader().readByte();
        const token = try allocator.alloc(u8, token_length);
        _ = try file.reader().read(token);
        self.vocab_map[i] = token;
    }

    self.init_ok = true;
    return self;
}

pub fn decode(self: Tokenizer, token: u32) []const u8 {
    if (self.init_ok and token < self.vocab_size) {
        return self.vocab_map[token];
    }
    return "";
}

pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
    if (self.init_ok) {
        for (self.vocab_map) |token| {
            allocator.free(token);
        }
        allocator.free(self.vocab_map);
    }
}
