//! HIP graphs: captured from a stream or built node by node, instantiated once, replayed and updated in place.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");

pub const Node = abi.GraphNode;

/// Starts capturing `stream`; every launch on it is recorded until `endCapture`, nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) Error!void {
    try stream.d.check(stream.d.api.hipStreamBeginCapture(stream.handle, mode), "hipStreamBeginCapture");
}

/// Ends the capture; an invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) Error!Graph {
    var g: abi.Graph = null;
    try stream.d.check(stream.d.api.hipStreamEndCapture(stream.handle, &g), "hipStreamEndCapture");
    return .{ .d = stream.d, .handle = g };
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.hipStreamIsCapturing(stream.handle, &s), "hipStreamIsCapturing");
    return s;
}

fn nodeParams(f: Function, cfg: launch.Config, args: *launch.Args) Error!abi.KernelNodeParams {
    try cfg.validate();
    if (cfg.cooperative or args.overflow) return error.Invalid;
    return .{
        .block = cfg.block,
        .extra = null,
        .func = f.handle,
        .grid = cfg.grid,
        .params = args.pointers(),
        .shared_bytes = cfg.shared,
    };
}

pub const Graph = struct {
    d: *const Driver,
    handle: abi.Graph,

    pub fn init(d: *const Driver) Error!Graph {
        var g: abi.Graph = null;
        try d.check(d.api.hipGraphCreate(&g, 0), "hipGraphCreate");
        return .{ .d = d, .handle = g };
    }

    pub fn deinit(self: *Graph) void {
        _ = self.d.api.hipGraphDestroy(self.handle);
        self.* = undefined;
    }

    /// A kernel node after `deps`; HIP copies the argument values, so `args` may change afterwards.
    pub fn addKernel(self: Graph, deps: []const Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!Node {
        const p = try nodeParams(f, cfg, args);
        var n: Node = null;
        try self.d.check(self.d.api.hipGraphAddKernelNode(&n, self.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p), "hipGraphAddKernelNode");
        return n;
    }

    /// New arguments or geometry for a node of this (template) graph, before it is instantiated again or updated from.
    pub fn setKernel(self: Graph, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.hipGraphKernelNodeSetParams(node, &p), "hipGraphKernelNodeSetParams");
    }

    pub fn depend(self: Graph, from: Node, to: Node) Error!void {
        const a = [1]Node{from};
        const b = [1]Node{to};
        try self.d.check(self.d.api.hipGraphAddDependencies(self.handle, &a, &b, 1), "hipGraphAddDependencies");
    }

    pub fn nodeCount(self: Graph) Error!usize {
        var n: usize = 0;
        try self.d.check(self.d.api.hipGraphGetNodes(self.handle, null, &n), "hipGraphGetNodes");
        return n;
    }

    /// Fills `out` with the graph's nodes (capture order for captured graphs) and returns them.
    pub fn nodes(self: Graph, out: []Node) Error![]Node {
        var n: usize = out.len;
        try self.d.check(self.d.api.hipGraphGetNodes(self.handle, out.ptr, &n), "hipGraphGetNodes");
        return out[0..@min(n, out.len)];
    }

    pub fn instantiate(self: Graph) Error!Exec {
        var e: abi.GraphExec = null;
        try self.d.check(self.d.api.hipGraphInstantiateWithFlags(&e, self.handle, 0), "hipGraphInstantiateWithFlags");
        return .{ .d = self.d, .handle = e };
    }
};

pub const Exec = struct {
    d: *const Driver,
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        _ = self.d.api.hipGraphExecDestroy(self.handle);
        self.* = undefined;
    }

    /// Moves the graph's work to the device ahead of the first launch, so that launch pays no setup.
    pub fn upload(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.hipGraphUpload(self.handle, stream.handle), "hipGraphUpload");
    }

    pub fn launchOn(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.hipGraphLaunch(self.handle, stream.handle), "hipGraphLaunch");
    }

    /// Changes one kernel node's arguments or geometry in the executable graph without rebuilding it.
    pub fn setKernel(self: Exec, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.hipGraphExecKernelNodeSetParams(self.handle, node, &p), "hipGraphExecKernelNodeSetParams");
    }

    /// Takes every node's parameters from `g`, which must have the same topology; returns HIP's verdict.
    pub fn update(self: Exec, g: Graph) Error!abi.ExecUpdateResult {
        var node: Node = null;
        var result: abi.ExecUpdateResult = .success;
        const res = self.d.api.hipGraphExecUpdate(self.handle, g.handle, &node, &result);
        if (res == abi.success) return result;
        if (result != .success) return result;
        try self.d.check(res, "hipGraphExecUpdate");
        unreachable;
    }
};
