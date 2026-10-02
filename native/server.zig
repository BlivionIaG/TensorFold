const std = @import("std");
const responses = @import("responses.zig");
const background = @import("background.zig");
const mx = @import("mlx.zig");
const inference = @import("session.zig");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const tool_calls = @import("tool_calls.zig");
const Request = std.http.Server.Request;
const control = @import("server_control.zig");
const memory_policy = @import("memory_budget.zig");
const memory_runtime = @import("memory_runtime.zig");
const PrefixStore = @import("prompt_cache.zig").Store(inference.Snapshot);
const live_status = @import("server_live.zig");
const shared_round = @import("shared_round.zig");

pub fn run(init: std.process.Init, args: []const []const u8) !void {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var limit: usize = 0;
    var timeout_ms: i64 = 0;
    var shutdown_grace_ms: i64 = 5000;
    var batch_streams: usize = 4;
    var batch_rows: usize = 128;
    var checkpoint_slots: ?usize = null;
    var prompt_cache_bytes: ?u64 = null;
    var snapshot_dir: ?[]const u8 = init.environ_map.get("TENSORFOLD_SNAPSHOT_DIR");
    var spill_bytes: u64 = 0;
    var max_snapshots: usize = 3;
    var name = std.fs.path.basename(args[2]);
    var defaults = try inference.Options.load(init.gpa, init.io, args[2]);
    defaults.max_tokens = 4096;
    var thinking = true;
    var vision_urls = false;
    var drafts = true;
    var draft_options = @import("neural_draft.zig").Options{ .enabled = true };
    var effort: ?[]const u8 = null;
    var overrides = std.json.Value{ .object = .empty };
    defer overrides.object.deinit(init.gpa);
    var i: usize = 3;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--no-drafts")) {
            drafts = false;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--vision-urls") or std.mem.eql(u8, args[i], "--no-vision-urls")) {
            vision_urls = std.mem.eql(u8, args[i], "--vision-urls");
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--thinking") or std.mem.eql(u8, args[i], "--no-thinking")) {
            thinking = std.mem.eql(u8, args[i], "--thinking");
            i += 1;
            continue;
        }
        if (i + 1 == args.len) return error.MissingArgument;
        const flag = args[i];
        const value = args[i + 1];
        i += 2;
        if (std.mem.eql(u8, flag, "--snapshot-dir")) {
            snapshot_dir = value;
            continue;
        }
        if (std.mem.eql(u8, flag, "--max-snapshots")) {
            max_snapshots = try std.fmt.parseInt(usize, value, 10);
            continue;
        }
        if (std.mem.eql(u8, flag, "--spill-gib")) {
            const size = try std.fmt.parseFloat(f64, value);
            if (!std.math.isFinite(size) or size < 0 or size >= 17179869184) return error.InvalidPromptCacheBudget;
            spill_bytes = @intFromFloat(size * memory_policy.gib);
            continue;
        }
        if (std.mem.eql(u8, flag, "--drafter")) {
            draft_options.directory = value;
            continue;
        }
        if (std.mem.eql(u8, flag, "--drafter-bits")) {
            draft_options.bits = try std.fmt.parseInt(i32, value, 10);
            continue;
        }
        if (std.mem.eql(u8, flag, "--max-draft") or std.mem.eql(u8, flag, "--mtp-drafts")) {
            draft_options.max_draft = try std.fmt.parseInt(usize, value, 10);
            continue;
        }
        if (std.mem.eql(u8, flag, "--draft-calibration")) {
            draft_options.calibration = value;
            continue;
        }
        if (std.mem.eql(u8, flag, "--batch-streams")) {
            batch_streams = try std.fmt.parseInt(usize, value, 10);
            if (batch_streams < 1 or batch_streams > shared_round.max_streams) return error.InvalidBatchStreams;
            continue;
        }
        if (std.mem.eql(u8, flag, "--batch-rows")) {
            batch_rows = try std.fmt.parseInt(usize, value, 10);
            if (batch_rows < 1 or batch_rows > 128) return error.InvalidBatchRows;
            continue;
        }
        if (std.mem.eql(u8, flag, "--checkpoint-slots")) {
            checkpoint_slots = try std.fmt.parseInt(usize, value, 10);
            if (checkpoint_slots.? == 0) return error.InvalidPromptCacheBudget;
            continue;
        }
        if (std.mem.eql(u8, flag, "--prompt-cache-gib")) {
            const size = try std.fmt.parseFloat(f64, value);
            if (!std.math.isFinite(size) or size < 0 or size >= 17179869184) return error.InvalidPromptCacheBudget;
            prompt_cache_bytes = @intFromFloat(size * memory_policy.gib);
            continue;
        }
        if (std.mem.eql(u8, flag, "--request-timeout-seconds")) timeout_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--shutdown-grace-seconds")) shutdown_grace_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--host")) host = value else if (std.mem.eql(u8, flag, "--port")) port = try std.fmt.parseInt(u16, value, 10) else if (std.mem.eql(u8, flag, "--served-model-name")) name = value else if (std.mem.eql(u8, flag, "--max-requests")) limit = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--reasoning-effort")) {
            if (!std.mem.eql(u8, value, "low") and !std.mem.eql(u8, value, "medium") and !std.mem.eql(u8, value, "xhigh")) return error.InvalidReasoningEffort;
            effort = value;
        } else {
            const fields = .{ .{ "--temperature", "temperature" }, .{ "--top-k", "top_k" }, .{ "--top-p", "top_p" }, .{ "--min-p", "min_p" }, .{ "--max-tokens", "max_tokens" }, .{ "--thinking-budget", "thinking_budget" } };
            var found = false;
            inline for (fields) |pair| if (std.mem.eql(u8, flag, pair[0])) {
                try overrides.object.put(init.gpa, pair[1], .{ .string = value });
                found = true;
            };
            if (!found) return error.UnknownArgument;
        }
    }
    defaults = try inference.Options.parseWithDefaults(init.gpa, overrides, defaults);
    draft_options.enabled = drafts and draft_options.max_draft > 0;
    try draft_options.validate();
    var signals = control.Signals.install();
    defer signals.deinit();
    var registry = control.Registry{ .io = init.io, .timeout_ms = timeout_ms, .shutdown_grace_ms = shutdown_grace_ms, .client_limit = 2 * @max(8, batch_streams) + 16 };
    const monitor = try std.Thread.spawn(.{}, control.Registry.watch, .{&registry});
    defer {
        registry.finished.store(true, .release);
        monitor.join();
    }
    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = try address.listen(init.io, .{ .kernel_backlog = 128 });
    defer listener.deinit(init.io);
    const path = try std.fmt.allocPrint(init.gpa, "{s}/config.json", .{args[2]});
    defer init.gpa.free(path);
    const config = try @import("weights.zig").readFile(init.io, path);
    defer mx.allocator.free(config);
    const parsed = try std.json.parseFromSlice(std.json.Value, init.gpa, config, .{});
    defer parsed.deinit();
    const model_type = if (parsed.value == .object) parsed.value.object.get("model_type") orelse std.json.Value.null else std.json.Value.null;
    const is_glm = model_type == .string and std.mem.eql(u8, model_type.string, "glm5_next");
    const is_flash = model_type == .string and std.mem.eql(u8, model_type.string, "qwen4_exp");
    if (is_glm) try @import("glm.zig").Model.prepareRuntime();
    var stats = live_status.Stats{ .io = init.io, .allocator = init.gpa };
    defer stats.deinit();
    var display = live_status.Display{ .io = init.io, .stats = &stats };
    var response_store = responses.Store{ .a = init.gpa };
    defer response_store.deinit();
    var worker = Worker{ .io = init.io, .dir = args[2], .defaults = defaults, .thinking = thinking, .effort = effort, .vision_urls = vision_urls, .control = &registry, .batch_streams = batch_streams, .is_glm = is_glm, .is_flash = is_flash, .memory_limit = init.environ_map.get("TENSORFOLD_MEMORY_LIMIT_GB"), .stats = &stats, .display = &display, .response_store = &response_store };
    worker.batch_rows = batch_rows;
    worker.queue.waiting_limit = @max(8, batch_streams);
    worker.checkpoint_slots = checkpoint_slots orelse @max(8, 3 * batch_streams);
    worker.prompt_cache_bytes = prompt_cache_bytes;
    const snapshot_path = if (snapshot_dir) |dir| if (std.ascii.eqlIgnoreCase(dir, "none")) null else try init.gpa.dupe(u8, dir) else if (init.environ_map.get("HOME")) |home| try std.fs.path.join(init.gpa, &.{ home, ".cache", "tensorfold", "native-prefix-snapshots" }) else null;
    defer if (snapshot_path) |path_value| init.gpa.free(path_value);
    worker.snapshot_dir = snapshot_path;
    worker.spill_bytes = spill_bytes;
    worker.max_snapshots = max_snapshots;
    worker.drafts = drafts;
    worker.draft_options = draft_options;
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    defer {
        worker.queue.close(init.io);
        thread.join();
    }
    worker.ready.waitUncancelable(init.io);
    if (worker.startup_error) |err| return err;
    if (registry.stopping.load(.acquire)) return;
    display.start(init.environ_map.get("TENSORFOLD_NO_LIVE"), init.environ_map.get("COLUMNS"));
    defer display.stop();
    var clients: std.Io.Group = .init;
    defer clients.await(init.io) catch clients.cancel(init.io);
    errdefer registry.stop();
    display.print("Native inference listening at http://{s}:{d} ({s})\n", .{ host, listener.socket.address.getPort(), name });
    const Event = union(enum) { accepted: anyerror!void, stopped: anyerror!void };
    var events: [2]Event = undefined;
    var select = std.Io.Select(Event).init(init.io, &events);
    defer select.cancelDiscard();
    try select.concurrent(.accepted, acceptRequests, .{ &worker, &listener, &clients, init.gpa, name, limit });
    try select.concurrent(.stopped, control.Registry.waitStopped, .{&registry});
    switch (try select.await()) {
        inline else => |result| try result,
    }
}

fn acceptRequests(worker: *Worker, listener: *std.Io.net.Server, clients: *std.Io.Group, a: std.mem.Allocator, name: []const u8, limit: usize) anyerror!void {
    var handled: usize = 0;
    while (limit == 0 or handled < limit) : (handled += 1) {
        const stream = try listener.accept(worker.io);
        const client = worker.control.acquire(stream.socket.handle) orelse {
            stream.close(worker.io);
            if (worker.control.stopping.load(.acquire)) return;
            continue;
        };
        clients.concurrent(worker.io, connectionTask, .{ worker, a, stream, name, handled, client }) catch |err| {
            worker.control.release(client);
            stream.close(worker.io);
            return err;
        };
    }
}

const Job = struct {
    stats: *live_status.Stats,
    activated: bool = false,
    a: std.mem.Allocator,
    request: ?*Request = null,
    model: []const u8,
    sequence: usize,
    created: i64,
    client: ?*control.Client = null,
    body: std.json.Value,
    is_chat: bool,
    options: inference.Options,
    response_reply: ?*responses.Reply = null,
    background: bool = false,
    suspended: ?*Pending = null,
    done: std.Io.Event = .unset,
    failure: ?anyerror = null,
    warmer: ?*Warmer = null,

    fn cancellation(job: *Job) @import("cancellation.zig").Cancellation {
        if (job.warmer) |warmer| return .{ .context = warmer, .callback = Warmer.checkCancellation };
        return job.client.?.cancellation();
    }

    fn finish(job: *Job, io: std.Io) void {
        if (job.warmer) |warmer| {
            warmer.inflight = false;
            warmer.index = if (job.failure == null) warmer.index + 1 else warmer.jobs.stops.len;
            if (warmer.index == warmer.jobs.stops.len) warmer.worker.warming.store(false, .release);
        } else job.done.set(io);
    }

    fn reportError(job: *Job, err: anyerror) !void {
        if (job.warmer != null) {
            job.failure = err;
            @import("snapshot_store.zig").report("warming", err);
        } else try requestFailure(job.a, job.request.?, err);
    }

    fn cancelled(job: *Job) bool {
        job.cancellation().check() catch return true;
        return false;
    }
};

const Warmer = struct {
    worker: *Worker,
    jobs: @import("prefill_plan.zig").BlockJobs,
    index: usize = 0,
    inflight: bool = false,
    arena: std.heap.ArenaAllocator,
    job: Job = undefined,

    fn init(worker: *Worker, session: *inference.Session, disk: *@import("snapshot_store.zig").Store) !?*Warmer {
        if (disk.reads != 0) return null;
        const blocks = try disk.blocksToWarm();
        defer {
            for (blocks) |tokens| mx.allocator.free(tokens);
            mx.allocator.free(blocks);
        }
        if (blocks.len == 0) return null;
        const pad = try session.tokenizer.encode(mx.allocator, "\n");
        defer mx.allocator.free(pad);
        if (pad.len == 0) return error.EmptyWarmPadding;
        const jobs = try @import("prefill_plan.zig").BlockJobs.init(mx.allocator, try session.prefillPlan(), blocks[0], @intCast(pad[pad.len - 1]));
        errdefer jobs.deinit(mx.allocator);
        if (jobs.stops.len == 0) {
            jobs.deinit(mx.allocator);
            return null;
        }
        try session.validate(jobs.probe, .{ .max_tokens = 1 });
        const warmer = try mx.allocator.create(Warmer);
        warmer.* = .{ .worker = worker, .jobs = jobs, .arena = .init(mx.allocator) };
        worker.warming.store(true, .release);
        return warmer;
    }

    fn deinit(w: *Warmer) void {
        w.worker.warming.store(false, .release);
        w.jobs.deinit(mx.allocator);
        w.arena.deinit();
        mx.allocator.destroy(w);
    }

    fn checkCancellation(context: ?*anyopaque) !void {
        const w: *Warmer = @ptrCast(@alignCast(context.?));
        if (w.worker.control.stopping.load(.acquire)) return error.ServerStopping;
    }

    fn enqueue(w: *Warmer) void {
        if (w.inflight or w.index >= w.jobs.stops.len) return;
        if (w.worker.control.stopping.load(.acquire)) {
            w.worker.warming.store(false, .release);
            return;
        }
        _ = w.arena.reset(.retain_capacity);
        w.job = .{ .stats = w.worker.stats, .a = w.arena.allocator(), .model = "", .sequence = 0, .created = 0, .body = .null, .is_chat = true, .options = .{ .max_tokens = 1, .draft = false, .sampling = .{ .temperature = 0 } }, .background = true, .warmer = w };
        w.inflight = w.worker.queue.put(w.worker.io, &w.job, false);
    }
};
const Worker = struct {
    response_store: *responses.Store,
    draft_options: @import("neural_draft.zig").Options = .{},
    drafts: bool = true,
    stats: *live_status.Stats,
    display: *live_status.Display,
    io: std.Io,
    dir: []const u8,
    queue: background.Queue(*Job, shared_round.max_streams) = .{},
    preemptions: std.atomic.Value(u64) = .init(0),
    warming: std.atomic.Value(bool) = .init(false),
    ready: std.Io.Event = .unset,
    startup_error: ?anyerror = null,
    control: *control.Registry,
    defaults: inference.Options,
    thinking: bool,
    effort: ?[]const u8,
    vision_urls: bool,
    batch_streams: usize,
    batch_rows: usize = 128,
    shared_decode: bool = false,
    is_glm: bool,
    is_flash: bool,
    memory_limit: ?[]const u8,
    memory_stats: MemoryStats = .{},
    checkpoint_slots: usize = 8,
    prompt_cache_bytes: ?u64 = null,
    snapshot_dir: ?[]const u8 = null,
    spill_bytes: u64 = 0,
    max_snapshots: usize = 3,
    disk: ?*@import("snapshot_store.zig").Store = null,
    prefix_stats: PrefixStats = .{},
    fn run(w: *Worker) void {
        w.loop() catch |err| {
            w.startup_error = err;
            w.ready.set(w.io);
        };
    }
    fn loop(w: *Worker) !void {
        try mx.init();
        defer mx.shutdown();
        var memory = try memory_runtime.Runtime.init(w.memory_limit);
        defer memory.deinit();
        try memory.checkWeightsAndDraft(w.io, w.dir, w.is_flash, if (w.draft_options.enabled) w.draft_options.directory else null);
        var session = try inference.Session.initWithDraft(w.io, w.dir, w.draft_options);
        defer session.deinit();
        if (w.batch_streams > shared_round.streamLimit(session.backend)) return error.InvalidBatchStreams;
        const shared_limit = shared_round.rowLimit(session.backend);
        w.shared_decode = shared_limit != 0;
        if (w.shared_decode) w.batch_rows = @min(w.batch_rows, shared_limit);
        var profile = try memory_runtime.measure(&session);
        var coordinator = shared_round.Coordinator{ .max_rows = w.batch_rows };
        try coordinator.calibrate(&session, w.batch_streams);
        profile.round_bytes = @max(profile.round_bytes, coordinator.peak_bytes);
        try memory.wire();
        var admission = memory_policy.Admission{ .budget = try memory.admissionBudget(w.io, @as(std.meta.Tag(inference.Backend), session.backend)), .solo_budget = memory.share, .memory = profile };
        var gate = memory_policy.StreamGate{ .budget = admission.budget, .per_token = profile.per_token, .work = profile.round_bytes };
        const prefix_budget = try admission.idleCacheBudget(@min(memory.ram / 8, 16 * memory_policy.gib), w.prompt_cache_bytes, try memory_runtime.activeBytes(), inference.context_limit, w.batch_streams);
        var prefixes: ?PrefixStore = if (prefix_budget == 0) null else try PrefixStore.init(mx.allocator, w.checkpoint_slots, prefix_budget);
        defer if (prefixes) |*store| store.deinit();
        if (prefixes) |*store| store.admit_oversize = true;
        var disk: ?@import("snapshot_store.zig").Store = null;
        defer if (disk) |*value| value.deinit();
        defer w.disk = null;
        if (prefixes) |*store| if (w.snapshot_dir) |directory| {
            disk = @import("snapshot_store.zig").Store.init(&session, directory, admission.budget, w.spill_bytes, w.max_snapshots) catch |err| blk: {
                @import("snapshot_store.zig").report("initialization", err);
                break :blk null;
            };
            if (disk) |*value| {
                w.disk = value;
                store.pinned_slots = @max(3, w.max_snapshots);
                store.eviction_context = value;
                store.on_evict = @import("snapshot_store.zig").Store.evicted;
                value.startup(store, @as(std.meta.Tag(inference.Backend), session.backend), try admission.projected(0, 64, 64, &.{})) catch |err| @import("snapshot_store.zig").report("startup", err);
            }
        };
        defer if (disk) |*value| if (prefixes) |*store| value.shutdown(store);
        const warmer = if (disk) |*value| Warmer.init(w, &session, value) catch |err| blk: {
            @import("snapshot_store.zig").report("warming setup", err);
            break :blk null;
        } else null;
        defer if (warmer) |value| value.deinit();
        w.prefix_stats.update(if (prefixes) |*store| store else null);
        w.memory_stats.budget = memory.budget;
        w.memory_stats.mlx_budget = memory.share;
        w.memory_stats.admission_budget = admission.budget;
        w.memory_stats.update(0);
        std.debug.print("Native memory admission: {d} bytes available, {d} resident, {d} bytes per token; stream={d}, decode work={d}, prefill work={d}\n", .{ admission.budget, try memory_runtime.activeBytes(), profile.per_token, try profile.streamBytes(64), profile.round_bytes, try profile.prefillBytes(profile.chunk) });
        w.ready.set(w.io);
        var active: [shared_round.max_streams]?*Pending = @splat(null);
        var live: usize = 0;
        var closed = false;
        var activation_order: u64 = 0;
        var decode_order: u64 = 1;
        var fill_schedule = FillSchedule{};
        while (!closed or live > 0) {
            if (warmer) |value| value.enqueue();
            while (w.queue.removeIf(w.io, Job.cancelled)) |job| {
                job.cancellation().check() catch |err| {
                    if (job.suspended) |pending| {
                        pending.reportError(err) catch |write_err| {
                            job.failure = write_err;
                        };
                        pending.deinit();
                        job.suspended = null;
                    } else job.reportError(err) catch |write_err| {
                        job.failure = write_err;
                    };
                };
                job.finish(w.io);
            }
            if (!w.control.stopping.load(.acquire)) try w.preempt(&session, &admission, &active, &live, if (prefixes) |*store| store else null);
            while (!closed and live < w.batch_streams) {
                const job = (w.queue.take(w.io, live == 0, false) catch blk: {
                    closed = true;
                    break :blk null;
                }) orelse break;
                const pending = Pending.start(w, &session, job) catch |err| blk: {
                    job.failure = err;
                    break :blk null;
                };
                if (pending) |p| {
                    for (&active) |*slot| if (slot.* == null) {
                        slot.* = p;
                        live += 1;
                        break;
                    };
                } else job.finish(w.io);
            }
            try gateRound(&gate, &active, if (prefixes) |*store| store else null);
            for (&active) |*slot| if (slot.*) |pending| if (pending.growth == .ended) {
                pending.reportError(error.StreamGrowthExceedsMemoryBudget) catch |err| {
                    pending.job.failure = err;
                };
                const job = pending.job;
                pending.deinit();
                slot.* = null;
                live -= 1;
                job.finish(w.io);
                try mx.check(mx.c.mlx_clear_cache());
            };
            var admissions: [shared_round.max_streams]*Pending = undefined;
            var admitting: usize = 0;
            for (active) |slot| if (slot) |pending| if (pending.generation == null and pending.admission_failure == null) {
                admissions[admitting] = pending;
                admitting += 1;
            };
            admitWaiting(AdmissionContext{ .session = &session, .admission = &admission, .active = &active, .prefixes = if (prefixes) |*store| store else null }, admissions[0..admitting], &activation_order);
            const filling = nextFill(&active);
            var decoding = false;
            for (active) |slot| if (slot) |pending| if (pending.generation) |*generation| {
                if (generation.isDecoding() and pending.growth == .run and pending.admission_failure == null and !Job.cancelled(pending.job)) decoding = true;
            };
            if (filling == null) fill_schedule = .{};
            const fill_due = filling != null and fill_schedule.due(decoding);
            var filled = false;
            var serial_decode_seconds: f64 = 0;
            var serial_decoded = false;
            var order: [shared_round.max_streams]usize = undefined;
            for (&order, 0..) |*slot, index| slot.* = index;
            if (filling) |chosen| for (active, 0..) |slot, index| {
                if (slot == chosen) {
                    std.mem.swap(usize, &order[0], &order[index]);
                    break;
                }
            };
            var candidates: [shared_round.max_streams]shared_round.Candidate = undefined;
            var candidate_count: usize = 0;
            for (order) |slot_index| {
                const slot = &active[slot_index];
                const pending = slot.* orelse continue;
                const before_fill = if (pending.generation) |*g| g.progress().prefilled else 0;
                if (pending.admission_failure == null and !Job.cancelled(pending.job)) {
                    if (pending.generation == null) continue;
                    if (pending.options.max_tokens > 0 and !pending.generation.?.isDecoding() and (pending != filling or !fill_due)) continue;
                    if (filled and pending.generation.?.isDecoding()) continue;
                }
                if (w.shared_decode and pending.admission_failure == null and !Job.cancelled(pending.job) and pending.growth == .run) {
                    if (pending.generation) |*g| if (g.isDecoding()) {
                        candidates[candidate_count] = .{ .slot = slot_index, .served = pending.decode_order, .activated = pending.activation_order };
                        candidate_count += 1;
                        continue;
                    };
                }
                const was_decoding = pending.generation != null and pending.generation.?.isDecoding() and pending.growth == .run and pending.admission_failure == null and !Job.cancelled(pending.job);
                const started = live_status.now(w.io);
                const done = pending.advance(&session, &admission, &active, if (prefixes) |*store| store else null) catch |err| blk: {
                    switch (err) {
                        error.RequestCancelled, error.RequestTimedOut, error.ServerStopping, error.RequestExceedsMemoryBudget => {
                            if (pending.generation != null) if (prefixes) |*store| pending.savePrefix(store, admission, &active) catch {};
                        },
                        else => {},
                    }
                    pending.reportError(err) catch |write_err| {
                        pending.job.failure = write_err;
                    };
                    break :blk true;
                };
                const elapsed = live_status.now(w.io) - started;
                if (pending.generation) |*g| if (g.progress().prefilled > before_fill) {
                    noteFill(&active, pending);
                    fill_schedule.filled(elapsed);
                    filled = true;
                };
                if (was_decoding) {
                    serial_decode_seconds += elapsed;
                    serial_decoded = true;
                }
                w.prefix_stats.update(if (prefixes) |*store| store else null);
                if (done) {
                    const job = pending.job;
                    pending.deinit();
                    slot.* = null;
                    live -= 1;
                    job.finish(w.io);
                }
            }
            if (serial_decoded) fill_schedule.decoded(serial_decode_seconds);
            const selected = shared_round.select(candidates[0..candidate_count], w.batch_rows);
            if (selected.len > 0) {
                var before: [shared_round.max_streams]inference.RequestGeneration.Progress = undefined;
                var results: [shared_round.max_streams]shared_round.Result = undefined;
                for (selected, 0..) |candidate, i| {
                    const pending = active[candidate.slot].?;
                    before[i] = pending.generation.?.progress();
                    pending.decode_order = decode_order;
                }
                decode_order +|= 1;
                const started = live_status.now(w.io);
                switch (session.backend) {
                    inline .qwen, .gemma, .nemotron, .flash => |*m, tag| {
                        var requests: [shared_round.max_streams]*inference.Generation(@TypeOf(m.*)) = undefined;
                        for (selected, 0..) |candidate, i| requests[i] = &@field(active[candidate.slot].?.generation.?, @tagName(tag));
                        coordinator.step(m, requests[0..selected.len], results[0..selected.len]) catch |err| {
                            for (results[0..selected.len]) |*result| result.* = .{ .failure = err };
                        };
                    },
                    else => unreachable,
                }
                const ended = live_status.now(w.io);
                fill_schedule.decoded(ended - started);
                w.stats.recordShared(coordinator.streams, coordinator.rows, coordinator.timing);
                var decoded: usize = 0;
                for (selected, 0..) |candidate, i| {
                    const pending = active[candidate.slot].?;
                    const after = pending.generation.?.progress();
                    decoded += after.decoded - before[i].decoded;
                    pending.recordDraftProgress(before[i], after);
                    var done = results[i].done;
                    var request_error = results[i].failure;
                    if (request_error == null and done) _ = pending.finishReply(&session) catch |err| {
                        request_error = err;
                    };
                    if (request_error) |err| {
                        pending.reportError(err) catch |write_err| {
                            pending.job.failure = write_err;
                        };
                        done = true;
                    }
                    if (done) {
                        const job = pending.job;
                        pending.deinit();
                        active[candidate.slot] = null;
                        live -= 1;
                        job.finish(w.io);
                    }
                }
                w.stats.record(0, decoded, started, ended);
                w.prefix_stats.update(if (prefixes) |*store| store else null);
            }
            var waiting: u64 = 0;
            for (active) |slot| if (slot) |pending| {
                if (pending.generation == null or pending.growth == .paused) waiting += 1;
            };
            w.memory_stats.growth_waits.store(gate.waits, .release);
            w.memory_stats.growth_ends.store(gate.ends, .release);
            w.memory_stats.update(waiting);
        }
    }

    fn preempt(w: *Worker, session: *inference.Session, admission: *memory_policy.Admission, active: []?*Pending, live: *usize, prefixes: ?*PrefixStore) !void {
        var waiting: ?*Pending = null;
        for (active) |slot| if (slot) |pending| if (!pending.job.background and pending.generation == null and !Job.cancelled(pending.job)) {
            if (waiting == null or pending.job.sequence < waiting.?.job.sequence) waiting = pending;
        };
        const queued = w.queue.foreground(w.io);
        if (waiting == null and queued == null) return;
        while (true) {
            const fits = if (waiting) |pending| (pending.reservePrefix(session, admission, active, prefixes) catch break) != null else live.* < w.batch_streams;
            if (fits) break;
            var victim: ?usize = null;
            for (active, 0..) |slot, i| if (slot) |pending| {
                if (!pending.job.background or Job.cancelled(pending.job)) continue;
                if (victim == null or backgroundBefore(pending, active[victim.?].?)) victim = i;
            };
            const at = victim orelse break;
            const pending = active[at].?;
            if (pending.generation != null) {
                if (prefixes) |store| pending.savePrefix(store, admission.*, active) catch {};
                _ = w.preemptions.fetchAdd(1, .release);
            }
            const job = pending.job;
            pending.preempt() catch |err| {
                pending.reportError(err) catch |write_err| {
                    job.failure = write_err;
                };
                pending.deinit();
                active[at] = null;
                live.* -= 1;
                job.finish(w.io);
                continue;
            };
            active[at] = null;
            live.* -= 1;
            job.suspended = pending;
            if (!w.queue.put(w.io, job, true)) {
                pending.reportError(error.ServerStopping) catch |err| {
                    job.failure = err;
                };
                pending.deinit();
                job.suspended = null;
                job.finish(w.io);
            }
            try mx.check(mx.c.mlx_clear_cache());
        }
    }
};

const fill_guard = 8;

const FillSchedule = struct {
    credit: f64 = 0,
    rounds_left: usize = 0,

    fn due(s: FillSchedule, decoding: bool) bool {
        return !decoding or s.credit <= 0 or s.rounds_left == 0;
    }

    fn filled(s: *FillSchedule, seconds: f64) void {
        s.credit = @min(s.credit, 0) + 0.25 * seconds;
        s.rounds_left = 32;
    }

    fn decoded(s: *FillSchedule, seconds: f64) void {
        s.credit -= seconds;
        s.rounds_left -|= 1;
    }
};

const AdmissionContext = struct {
    session: *inference.Session,
    admission: *memory_policy.Admission,
    active: []const ?*Pending,
    prefixes: ?*PrefixStore,

    fn admit(context: AdmissionContext, pending: *Pending) !bool {
        return pending.admit(context.session, context.admission, context.active, context.prefixes);
    }
};

fn admissionBefore(_: void, lhs: *Pending, rhs: *Pending) bool {
    if (lhs.job.background != rhs.job.background) return !lhs.job.background;
    return lhs.job.sequence < rhs.job.sequence;
}

fn admitWaiting(context: anytype, waiting: []*Pending, activation_order: *u64) void {
    std.mem.sort(*Pending, waiting, {}, admissionBefore);
    for (waiting) |pending| {
        const admitted = context.admit(pending) catch |err| {
            pending.admission_failure = err;
            continue;
        };
        if (!admitted) break;
        pending.activation_order = activation_order.*;
        activation_order.* += 1;
    }
}

test "admission preserves foreground arrival order and holds followers until memory is available" {
    const Probe = struct {
        calls: [3]usize = undefined,
        count: usize = 0,
        terminal: bool = false,

        fn admit(probe: *@This(), pending: *Pending) !bool {
            probe.calls[probe.count] = pending.job.sequence;
            probe.count += 1;
            if (pending.job.sequence == 2) {
                if (probe.terminal) return error.RequestExceedsMemoryBudget;
                return false;
            }
            return true;
        }
    };
    var jobs: [3]Job = undefined;
    var pending: [3]Pending = undefined;
    var waiting: [3]*Pending = undefined;
    for (&jobs, &pending, &waiting, 0..) |*job, *p, *slot, i| {
        job.sequence = i + 1;
        job.background = i == 0;
        p.* = .{ .job = job };
        slot.* = p;
    }
    var probe = Probe{};
    var order: u64 = 9;
    admitWaiting(&probe, &waiting, &order);
    try std.testing.expectEqualSlices(usize, &.{2}, probe.calls[0..probe.count]);
    try std.testing.expectEqual(@as(u64, 9), order);
    try std.testing.expect(pending[1].admission_failure == null);
    probe = .{ .terminal = true };
    admitWaiting(&probe, &waiting, &order);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 1 }, probe.calls[0..probe.count]);
    try std.testing.expectEqual(error.RequestExceedsMemoryBudget, pending[1].admission_failure.?);
    try std.testing.expectEqual(@as(u64, 9), pending[2].activation_order);
    try std.testing.expectEqual(@as(u64, 10), pending[0].activation_order);
}

test "prompt chunks lend decode time with bounded rounds and carry excess decode debt" {
    var schedule = FillSchedule{};
    try std.testing.expect(schedule.due(true));
    schedule.filled(4);
    try std.testing.expect(!schedule.due(true));
    try std.testing.expect(schedule.due(false));
    schedule.decoded(0.75);
    try std.testing.expect(!schedule.due(true));
    schedule.decoded(0.5);
    try std.testing.expect(schedule.due(true));
    schedule.filled(0.5);
    try std.testing.expectEqual(@as(f64, -0.125), schedule.credit);
    try std.testing.expect(schedule.due(true));
    schedule.filled(1);
    try std.testing.expectEqual(@as(f64, 0.125), schedule.credit);
    schedule.filled(4);
    try std.testing.expectEqual(@as(f64, 1), schedule.credit);
    for (0..31) |_| schedule.decoded(0);
    try std.testing.expect(!schedule.due(true));
    schedule.decoded(0);
    try std.testing.expect(schedule.due(true));
}

test "singleton admission reserves replies beyond the concurrent growth horizon" {
    const tokens = [_]i32{ 10, 11, 12, 13 };
    const concurrent = Pending{ .job = undefined, .ids = &tokens, .options = .{ .max_tokens = 8192 } };
    const singleton = Pending{ .job = undefined, .ids = &tokens, .options = .{ .max_tokens = 8192 }, .reservation_horizon = null };
    const stream = memory_policy.Live{ .now = 4, .most = 8196, .copy_bytes = 17 };
    try std.testing.expectEqual(@as(u64, 2048), concurrent.reservedReply());
    try std.testing.expectEqual(@as(u64, 8192), singleton.reservedReply());
    try std.testing.expectEqualDeep(memory_policy.Live{ .now = 4, .most = 2052, .copy_bytes = 17 }, concurrent.reserve(stream));
    try std.testing.expectEqualDeep(stream, singleton.reserve(stream));
    var admission = memory_policy.Admission{
        .budget = 4096,
        .memory = .{ .short_tokens = 1, .short = 1, .long_tokens = 2, .long = 2, .per_token = 1, .prefill_a = 0, .prefill_b = 0, .round_bytes = 0 },
    };
    try std.testing.expect(try admission.admits(0, tokens.len, tokens.len + concurrent.reservedReply(), &.{}));
    try std.testing.expect(!try admission.admits(0, tokens.len, tokens.len + singleton.reservedReply(), &.{}));
}

test "sole admission and prefill retain the full measured workspace within the MLX share" {
    var admission = memory_policy.Admission{
        .budget = 1149,
        .solo_budget = 1500,
        .memory = .{ .short_tokens = 1000, .short = 100, .long_tokens = 2000, .long = 100, .per_token = 0, .prefill_a = 0, .prefill_b = 0, .round_bytes = 50 },
    };
    var pending = Pending{ .job = undefined, .options = .{ .max_tokens = 32 } };
    var waiting = Pending{ .job = undefined, .options = .{ .max_tokens = 32 } };
    const active = [_]?*Pending{ &pending, &waiting };
    const prefix = try admission.prefixProjected(1000, 8, 40, &.{}, 0, 0, 0);
    try std.testing.expectEqual(@as(u64, 1150), prefix.take);
    try std.testing.expect(prefix.take > admission.budget);
    try std.testing.expect(prefix.take <= pending.requestBudget(admission, &active));

    pending.generation = .{ .qwen = undefined };
    const filling = try admission.fillingProjected(1100, 8, .{ .now = 8, .most = 40 }, &.{});
    try std.testing.expectEqual(@as(u64, 1150), filling);
    try std.testing.expect(filling > admission.budget);
    try std.testing.expect(filling <= pending.requestBudget(admission, &active));
    try std.testing.expectEqual(@as(u64, 50), try admission.roundBytes(1));

    admission.solo_budget = 1150;
    try std.testing.expect(prefix.take <= pending.requestBudget(admission, &active));
    try std.testing.expect(filling <= pending.requestBudget(admission, &active));
    admission.solo_budget = 1149;
    try std.testing.expect(prefix.take > pending.requestBudget(admission, &active));
    try std.testing.expect(filling > pending.requestBudget(admission, &active));
    admission.solo_budget = null;
    try std.testing.expectEqual(admission.budget, pending.requestBudget(admission, &active));
}

test "live and paused peers retain the concurrent budget until their reservations retire" {
    const admission = memory_policy.Admission{
        .budget = 1149,
        .solo_budget = 1500,
        .memory = .{ .short_tokens = 1000, .short = 100, .long_tokens = 2000, .long = 100, .per_token = 0, .prefill_a = 0, .prefill_b = 0, .round_bytes = 50 },
    };
    var pending = Pending{ .job = undefined, .options = .{ .max_tokens = 32 } };
    var peer = Pending{ .job = undefined, .options = .{ .max_tokens = 32 }, .generation = .{ .qwen = undefined } };
    var waiting = Pending{ .job = undefined, .options = .{ .max_tokens = 32 } };
    var empty = Pending{ .job = undefined, .options = .{ .max_tokens = 0 }, .generation = .{ .qwen = undefined } };
    const active = [_]?*Pending{ &pending, &peer, &waiting, &empty, null };
    const live = [_]memory_policy.Live{.{ .now = 8, .most = 40 }};
    const prefix = try admission.prefixProjected(1100, 8, 40, &live, 0, 0, 0);
    const filling = try admission.fillingProjected(1200, 8, .{ .now = 8, .most = 40 }, &live);
    for ([_]bool{ false, true }) |paused| {
        peer.growth = if (paused) .paused else .run;
        try std.testing.expectEqual(admission.budget, pending.requestBudget(admission, &active));
        try std.testing.expect(prefix.take > pending.requestBudget(admission, &active));
        pending.generation = .{ .qwen = undefined };
        try std.testing.expectEqual(admission.budget, pending.requestBudget(admission, &active));
        try std.testing.expect(filling > pending.requestBudget(admission, &active));
        pending.generation = null;
    }
    peer.generation = null;
    const alone = try admission.prefixProjected(1000, 8, 40, &.{}, 0, 0, 0);
    try std.testing.expect(alone.take <= pending.requestBudget(admission, &active));
    pending.generation = .{ .qwen = undefined };
    try std.testing.expectEqual(@as(u64, 1500), pending.requestBudget(admission, &active));
    try std.testing.expect((try admission.fillingProjected(1100, 8, .{ .now = 8, .most = 40 }, &.{})) <= pending.requestBudget(admission, &active));
}

fn backgroundBefore(lhs: *const Pending, rhs: *const Pending) bool {
    const left_fill = lhs.generation == null or !lhs.generation.?.isDecoding();
    const right_fill = rhs.generation == null or !rhs.generation.?.isDecoding();
    if (left_fill != right_fill) return left_fill;
    return if (left_fill) lhs.job.sequence > rhs.job.sequence else lhs.activation_order < rhs.activation_order;
}

fn nextFill(active: []const ?*Pending) ?*Pending {
    var chosen: ?*Pending = null;
    for (active) |slot| if (slot) |pending| {
        if (pending.generation == null or pending.options.max_tokens == 0 or pending.generation.?.isDecoding() or pending.growth != .run or pending.admission_failure != null or Job.cancelled(pending.job)) continue;
        if (chosen == null or fillBefore(pending, chosen.?)) chosen = pending;
    };
    return chosen;
}

fn fillBefore(lhs: *const Pending, rhs: *const Pending) bool {
    const left_due = lhs.passed >= fill_guard;
    const right_due = rhs.passed >= fill_guard;
    if (left_due != right_due) return left_due;
    if (left_due) {
        if (lhs.passed != rhs.passed) return lhs.passed > rhs.passed;
    } else {
        if (lhs.job.background != rhs.job.background) return !lhs.job.background;
        const left = lhs.ids.len -| if (lhs.generation) |*g| g.progress().prefilled else 0;
        const right = rhs.ids.len -| if (rhs.generation) |*g| g.progress().prefilled else 0;
        if (left != right) return left < right;
    }
    return lhs.job.sequence < rhs.job.sequence;
}

fn noteFill(active: []const ?*Pending, selected: *Pending) void {
    for (active) |slot| if (slot) |pending| if (pending.generation != null and !pending.generation.?.isDecoding()) {
        pending.passed = if (pending == selected) 0 else pending.passed +| 1;
    };
    selected.passed = 0;
}

test "prompt priority is foreground then shortest with a bounded age guard" {
    var old_job: Job = undefined;
    old_job.sequence = 1;
    old_job.background = false;
    var new_job: Job = undefined;
    new_job.sequence = 9;
    new_job.background = false;
    const tokens: [40]i32 = @splat(0);
    var older = Pending{ .job = &old_job, .ids = &tokens };
    var shorter = Pending{ .job = &new_job, .ids = tokens[0..4] };
    for (0..fill_guard) |passed| {
        older.passed = passed;
        try std.testing.expect(fillBefore(&shorter, &older));
    }
    older.passed = fill_guard;
    try std.testing.expect(fillBefore(&older, &shorter));
    shorter.passed = fill_guard + 1;
    try std.testing.expect(fillBefore(&shorter, &older));
    shorter.passed = fill_guard;
    try std.testing.expect(fillBefore(&older, &shorter));
    older.passed = 0;
    shorter.passed = 0;
    new_job.background = true;
    try std.testing.expect(fillBefore(&older, &shorter));
    shorter.passed = fill_guard;
    try std.testing.expect(fillBefore(&shorter, &older));
    shorter.passed = 0;
    shorter.ids = &tokens;
    new_job.background = false;
    try std.testing.expect(fillBefore(&older, &shorter));
    old_job.background = true;
    new_job.background = true;
    try std.testing.expect(backgroundBefore(&shorter, &older));
}

fn gateRound(gate: *memory_policy.StreamGate, active: []const ?*Pending, prefixes: ?*PrefixStore) !void {
    var oldest: [shared_round.max_streams]*Pending = undefined;
    var count: usize = 0;
    var residents: usize = 0;
    for (active) |slot| if (slot != null) {
        residents += 1;
    };
    for (active) |slot| if (slot) |pending| {
        pending.growth = .run;
        if (residents > 1) if (pending.generation) |*generation| generation.discardPreview();
        if (pending.generation) |*generation| if (generation.isDecoding()) {
            oldest[count] = pending;
            count += 1;
        };
    };
    std.mem.sort(*Pending, oldest[0..count], {}, struct {
        fn less(_: void, lhs: *Pending, rhs: *Pending) bool {
            return lhs.activation_order < rhs.activation_order;
        }
    }.less);
    var streams: [shared_round.max_streams]memory_policy.Live = undefined;
    for (oldest[0..count], streams[0..count]) |pending, *stream| {
        stream.* = pending.generation.?.memoryLengths();
        stream.copy_bytes = pending.prefix_reserve;
    }
    const plan = try gate.plan(memory_runtime.Reclaim{ .prefixes = prefixes }, streams[0..count]);
    for (oldest[plan.run..count]) |pending| {
        pending.growth = .paused;
        pending.generation.?.discardPreview();
    }
    if (plan.ended) |index| oldest[index].growth = .ended;
}

pub fn checkGrowth(session: *inference.Session, profile: memory_policy.StreamMemory) !void {
    const tokens = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    for ([_]f32{ 0, 0.7 }) |temperature| {
        const options = inference.Options{ .max_tokens = 12, .ignore_eos = true, .sampling = .{ .temperature = temperature, .top_k = 0, .top_p = 1, .metal = true }, .seed = 71 };
        var isolated = try inference.RequestGeneration.init(session, mx.allocator, &tokens, options, .{}, null);
        defer isolated.deinit();
        while (!try isolated.step(session)) {}
        var expected = try isolated.takeReply();
        defer expected.deinit(mx.allocator);

        var older = Pending{ .job = undefined, .activation_order = 2 };
        defer if (older.generation) |*g| g.deinit();
        var newer = Pending{ .job = undefined, .activation_order = 9 };
        defer if (newer.generation) |*g| g.deinit();
        for ([_]*Pending{ &older, &newer }) |pending| {
            pending.generation = try inference.RequestGeneration.init(session, mx.allocator, &tokens, options, .{}, null);
            while (!pending.generation.?.isDecoding()) _ = try pending.generation.?.step(session);
        }
        // A reused worker slot can contain a younger stream than a later slot.
        const active = [_]?*Pending{ &newer, null, &older };
        var gate = memory_policy.StreamGate{ .budget = 0, .per_token = memory_policy.gib, .work = 0, .horizon = 1 };
        const memory = memory_runtime.Reclaim{ .prefixes = null };
        try mx.check(mx.c.mlx_clear_cache());
        gate.budget = try memory.used() + memory_policy.gib;
        const before = newer.generation.?.progress();
        try gateRound(&gate, &active, null);
        if (older.growth != .run or newer.growth != .paused) return error.GrowthActivationOrderMismatch;
        var registry = control.Registry{ .io = session.io };
        var client = control.Client{ .owner = &registry };
        var job: Job = undefined;
        job.client = &client;
        job.warmer = null;
        newer.job = &job;
        var admission = memory_policy.Admission{ .budget = gate.budget, .memory = .{ .short_tokens = 0, .short = 0, .long_tokens = 1, .long = 0, .per_token = 1, .prefill_a = 0, .prefill_b = 0, .round_bytes = 0 } };
        try std.testing.expectError(error.RequestCancelled, newer.advance(session, &admission, &active, null));
        client.deadline = 0;
        try std.testing.expectError(error.RequestTimedOut, newer.advance(session, &admission, &active, null));
        client = .{ .owner = &registry };
        registry.stop();
        try std.testing.expectError(error.ServerStopping, newer.advance(session, &admission, &active, null));
        _ = try older.generation.?.step(session);
        try std.testing.expectEqualDeep(before, newer.generation.?.progress());

        gate.budget = std.math.maxInt(u64);
        try gateRound(&gate, &active, null);
        if (older.growth != .run or newer.growth != .run) return error.GrowthBudgetRecoveryMismatch;
        try mx.check(mx.c.mlx_clear_cache());
        gate.per_token = 0;
        gate.budget = try memory.used() + memory_policy.gib;
        newer.prefix_reserve = 2 * memory_policy.gib;
        try gateRound(&gate, &active, null);
        if (older.growth != .run or newer.growth != .paused) return error.GrowthPrefixReservationMismatch;
        newer.prefix_reserve = 0;
        try gateRound(&gate, &active, null);
        if (older.growth != .run or newer.growth != .run) return error.GrowthPrefixRecoveryMismatch;
        for ([_]*Pending{ &older, &newer }) |pending| {
            while (!try pending.generation.?.step(session)) {}
            var actual = try pending.generation.?.takeReply();
            defer actual.deinit(mx.allocator);
            try std.testing.expectEqualSlices(u32, expected.tokens.items, actual.tokens.items);
            try std.testing.expectEqualStrings(expected.content, actual.content);
        }
        try std.testing.expectEqual(@as(u64, 2), gate.waits);
        try std.testing.expectEqual(@as(u64, 0), gate.ends);
    }

    var older = Pending{ .job = undefined, .activation_order = 3 };
    defer if (older.generation) |*g| g.deinit();
    var newer = Pending{ .job = undefined, .activation_order = 4 };
    defer if (newer.generation) |*g| g.deinit();
    for ([_]*Pending{ &older, &newer }) |pending| {
        pending.generation = try inference.RequestGeneration.init(session, mx.allocator, &tokens, .{ .max_tokens = 12, .ignore_eos = true }, .{}, null);
        while (!pending.generation.?.isDecoding()) _ = try pending.generation.?.step(session);
    }
    var gate = memory_policy.StreamGate{ .budget = 0, .per_token = 1, .work = 0 };
    try gateRound(&gate, &.{ &newer, &older }, null);
    if (older.growth != .run or newer.growth != .ended) return error.GrowthOldestSurvivorMismatch;
    newer.generation.?.deinit();
    newer.generation = null;
    try mx.check(mx.c.mlx_clear_cache());
    gate.budget = std.math.maxInt(u64);
    try gateRound(&gate, &.{&older}, null);
    while (!try older.generation.?.step(session)) {}
    try std.testing.expectEqual(@as(u64, 1), gate.ends);
    const prompt = try mx.allocator.alloc(i32, @intCast(profile.chunk + 1));
    defer mx.allocator.free(prompt);
    for (prompt, 0..) |*token, i| token.* = @intCast(10 + i % 93);
    var filling = Pending{ .job = undefined, .ids = prompt };
    filling.generation = try inference.RequestGeneration.init(session, mx.allocator, prompt, .{ .max_tokens = 4, .ignore_eos = true }, .{}, null);
    defer filling.generation.?.deinit();
    _ = try filling.generation.?.step(session);
    if (filling.generation.?.isDecoding()) return error.GrowthExpectedPartialPrefill;
    const before = filling.generation.?.progress();
    var registry = control.Registry{ .io = session.io };
    var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(session.io, .{});
    defer listener.deinit(session.io);
    const peer = try listener.socket.address.connect(session.io, .{ .mode = .stream });
    defer peer.close(session.io);
    const socket = try listener.accept(session.io);
    defer socket.close(session.io);
    var client = control.Client{ .owner = &registry, .socket = socket.socket.handle };
    try client.cancellation().check();
    var job: Job = undefined;
    job.client = &client;
    job.warmer = null;
    job.sequence = 1;
    job.background = false;
    filling.job = &job;
    const selected = @import("prompt_cache.zig").checkpoints(prompt.len - 1, 0, null, prompt).aligned(filling.generation.?.boundary(), 0, prompt.len);
    if (!selected.contains(before.prefilled)) return error.GrowthExpectedHistoryCheckpoint;
    var checkpoint = (try filling.generation.?.snapshot()).?;
    defer checkpoint.deinit();
    var admission = memory_policy.Admission{ .budget = 0, .memory = profile };
    try std.testing.expectError(error.RequestExceedsMemoryBudget, filling.guardPrefill(admission, &.{&filling}, null));
    try std.testing.expectEqualDeep(before, filling.generation.?.progress());
    {
        var live_reply = Pending{ .job = undefined, .ids = &tokens, .options = .{ .max_tokens = 12, .ignore_eos = true } };
        live_reply.generation = try inference.RequestGeneration.init(session, mx.allocator, &tokens, live_reply.options, .{}, null);
        defer live_reply.generation.?.deinit();
        while (!live_reply.generation.?.isDecoding()) _ = try live_reply.generation.?.step(session);
        if (try filling.guardPrefill(admission, &.{ &filling, &live_reply }, null)) return error.GrowthExpectedPrefillWait;
        try std.testing.expectEqualDeep(before, filling.generation.?.progress());
        var waiting = Pending{ .job = &job, .ids = prompt, .options = .{ .max_tokens = 4, .ignore_eos = true } };
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        const resident = try memory_runtime.activeBytes();
        if (try waiting.admit(session, &admission, &.{ &filling, &live_reply, &waiting }, null)) return error.GrowthExpectedAdmissionWait;
        if (waiting.generation != null) return error.GrowthWaitingRequestActivated;
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        const after_wait = try memory_runtime.activeBytes();
        if (after_wait != resident) {
            std.debug.print("Refused admission resident bytes: before={d}, after={d}\n", .{ resident, after_wait });
            return error.GrowthRefusedAdmissionResidencyMismatch;
        }
        var younger_job: Job = undefined;
        younger_job.client = &client;
        younger_job.warmer = null;
        younger_job.sequence = 2;
        younger_job.background = false;
        var younger = Pending{ .job = &younger_job, .ids = &tokens, .options = .{ .max_tokens = 0 } };
        const active = [_]?*Pending{ &filling, &live_reply, &waiting, &younger };
        if ((try younger.reservePrefix(session, &admission, &active, null)) == null) return error.GrowthExpectedAdmissibleFollower;
        var waiting_list = [_]*Pending{ &younger, &waiting };
        var activation_order: u64 = 7;
        admitWaiting(AdmissionContext{ .session = session, .admission = &admission, .active = &active, .prefixes = null }, &waiting_list, &activation_order);
        if (waiting.generation != null or younger.generation != null) return error.GrowthFollowerBypassedWaitingRequest;
        try std.testing.expectEqual(@as(u64, 7), activation_order);
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        const after_followers = try memory_runtime.activeBytes();
        if (after_followers != resident) {
            std.debug.print("Held followers resident bytes: before={d}, after={d}\n", .{ resident, after_followers });
            return error.GrowthHeldFollowersResidencyMismatch;
        }
        const reply_before = live_reply.generation.?.progress();
        const position_before = live_reply.generation.?.memoryLengths().now;
        _ = try live_reply.generation.?.step(session);
        if (live_reply.generation.?.memoryLengths().now <= position_before) return error.GrowthWaitingBlockedDecode;
        _ = try live_reply.generation.?.step(session);
        if (live_reply.generation.?.progress().decoded <= reply_before.decoded) return error.GrowthWaitingBlockedPublication;
        registry.stop();
        try std.testing.expectError(error.ServerStopping, waiting.admit(session, &admission, &.{ &filling, &live_reply, &waiting }, null));
        if (waiting.generation != null) return error.GrowthCancelledRequestActivated;
        try std.testing.expectError(error.ServerStopping, filling.advance(session, &admission, &.{ &filling, &live_reply }, null));
        try std.testing.expectEqualDeep(before, filling.generation.?.progress());
    }
    admission.budget = std.math.maxInt(u64);
    if (!try filling.guardPrefill(admission, &.{&filling}, null)) return error.GrowthExpectedPrefillRecovery;
    while (!try filling.generation.?.step(session)) {}
    var expected = try filling.generation.?.takeReply();
    defer expected.deinit(mx.allocator);
    var resumed = try inference.RequestGeneration.init(session, mx.allocator, prompt, .{ .max_tokens = 4, .ignore_eos = true }, .{}, null);
    defer resumed.deinit();
    try resumed.restorePrefix(&checkpoint);
    try std.testing.expectEqual(before.prefilled, resumed.progress().prefilled);
    while (!try resumed.step(session)) {}
    var actual = try resumed.takeReply();
    defer actual.deinit(mx.allocator);
    try std.testing.expectEqualSlices(u32, expected.tokens.items, actual.tokens.items);
    try std.testing.expectEqualStrings(expected.content, actual.content);
    for ([_]bool{ false, true }) |take| {
        var store = try PrefixStore.init(mx.allocator, 4, std.math.maxInt(u64));
        defer store.deinit();
        try store.insertOwned(prompt[0..before.prefilled], try checkpoint.clone(), prompt, false);
        var next = Pending{ .job = undefined, .ids = prompt, .options = .{ .max_tokens = 4, .ignore_eos = true } };
        admission.budget = 0;
        try std.testing.expectError(error.RequestExceedsMemoryBudget, next.reservePrefix(session, &admission, &.{}, &store));
        try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
        try std.testing.expectEqual(@as(u64, 0), store.evictions);
        admission.budget = std.math.maxInt(u64);
        const full = (try next.reservePrefix(session, &admission, &.{}, &store)).?;
        if (full.prefix.copy <= full.prefix.take) return error.GrowthExpectedPrefixCopyCost;
        const shared = (try next.reservePrefix(session, &admission, &.{&filling}, &store)).?;
        try std.testing.expectEqual(@min(checkpoint.nbytes(), filling.generation.?.cacheBytes()), shared.prefix.shared);
        if (shared.prefix.copy - shared.prefix.take > full.prefix.copy - full.prefix.take) return error.GrowthSharedPrefixReservationMismatch;
        if (take) {
            const unrelated = try mx.allocator.dupe(i32, prompt);
            defer mx.allocator.free(unrelated);
            unrelated[0] += 1;
            try store.insertOwned(unrelated[0..before.prefilled], try checkpoint.clone(), unrelated, false);
            admission.budget = full.prefix.take + (full.prefix.copy - full.prefix.take) / 2;
        }
        const reserved = (try next.reservePrefix(session, &admission, &.{}, &store)).?;
        try std.testing.expectEqual(take, reserved.prefix.copy > reserved.budget);
        try std.testing.expectEqual(admission.budget, reserved.budget);
        try std.testing.expectEqual(@as(u64, @intFromBool(take)), store.evictions);
        const pointer = switch (store.entries.items[0].cache) {
            inline else => |state| @intFromPtr(state.cache.ptr),
        };
        next.generation = try inference.RequestGeneration.init(session, mx.allocator, prompt, next.options, .{}, null);
        defer next.generation.?.deinit();
        try next.restorePrefix(&store, reserved);
        try std.testing.expectEqual(@as(usize, @intFromBool(!take)), store.entries.items.len);
        try std.testing.expectEqual(if (take) @as(u64, 0) else checkpoint.nbytes(), next.prefix_reserve);
        if (take) switch (next.generation.?) {
            inline else => |request| try std.testing.expectEqual(pointer, @intFromPtr(request.state.cache.ptr)),
        };
        while (!try next.generation.?.step(session)) {}
        var result = try next.generation.?.takeReply();
        defer result.deinit(mx.allocator);
        try std.testing.expectEqualSlices(u32, expected.tokens.items, result.tokens.items);
        try std.testing.expectEqualStrings(expected.content, result.content);
    }
    std.debug.print("PASS: live cache growth gating preserves activation order, paused greedy/seeded output and oldest-stream recovery\n", .{});
    std.debug.print("PASS: prefill admission preserves request state for subsequent completion\n", .{});
    std.debug.print("PASS: selected history checkpoint resumes with identical tokens and content\n", .{});
    std.debug.print("PASS: admission protects the matching prefix, takes ownership only under pressure, and preserves resumed output\n", .{});
}

const PrefixStats = struct {
    enabled: std.atomic.Value(bool) = .init(false),
    bytes: std.atomic.Value(u64) = .init(0),
    entries: std.atomic.Value(usize) = .init(0),
    hits: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),
    evictions: std.atomic.Value(u64) = .init(0),

    fn update(s: *PrefixStats, store: ?*const PrefixStore) void {
        if (store) |p| {
            s.enabled.store(true, .release);
            s.bytes.store(p.nbytes(), .release);
            s.entries.store(p.entries.items.len, .release);
            s.hits.store(p.hits, .release);
            s.misses.store(p.misses, .release);
            s.evictions.store(p.evictions, .release);
        }
    }

    fn snapshot(s: *const PrefixStats) struct { enabled: bool, bytes: u64, entries: usize, hits: u64, misses: u64, evictions: u64 } {
        return .{ .enabled = s.enabled.load(.acquire), .bytes = s.bytes.load(.acquire), .entries = s.entries.load(.acquire), .hits = s.hits.load(.acquire), .misses = s.misses.load(.acquire), .evictions = s.evictions.load(.acquire) };
    }
};

const MemoryStats = struct {
    budget: u64 = 0,
    mlx_budget: u64 = 0,
    admission_budget: u64 = 0,
    active: std.atomic.Value(usize) = .init(0),
    cache: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    waiting: std.atomic.Value(u64) = .init(0),
    growth_waits: std.atomic.Value(u64) = .init(0),
    growth_ends: std.atomic.Value(u64) = .init(0),

    fn update(stats: *MemoryStats, waiting: u64) void {
        var value: usize = 0;
        if (mx.c.mlx_get_active_memory(&value) == 0) stats.active.store(value, .release);
        if (mx.c.mlx_get_cache_memory(&value) == 0) stats.cache.store(value, .release);
        if (mx.c.mlx_get_peak_memory(&value) == 0) stats.peak.store(value, .release);
        stats.waiting.store(waiting, .release);
    }

    fn snapshot(stats: *const MemoryStats) struct { budget: u64, mlx_budget: u64, admission_budget: u64, active: usize, cache: usize, peak: usize, waiting_requests: u64, growth_waits: u64, growth_ends: u64, probe_repeats: u32 } {
        return .{ .budget = stats.budget, .mlx_budget = stats.mlx_budget, .admission_budget = stats.admission_budget, .active = stats.active.load(.acquire), .cache = stats.cache.load(.acquire), .peak = stats.peak.load(.acquire), .waiting_requests = stats.waiting.load(.acquire), .growth_waits = stats.growth_waits.load(.acquire), .growth_ends = stats.growth_ends.load(.acquire), .probe_repeats = memory_policy.probe_repeats };
    }
};
fn connectionTask(w: *Worker, a: std.mem.Allocator, stream: std.Io.net.Stream, name: []const u8, sequence: usize, client: *control.Client) void {
    defer stream.close(w.io);
    defer w.control.release(client);
    var input: [65536]u8 = undefined;
    var output: [8192]u8 = undefined;
    var reader = stream.reader(w.io, &input);
    var writer = stream.writer(w.io, &output);
    var http = std.http.Server.init(&reader.interface, &writer.interface);
    var request = http.receiveHead() catch return;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    handle(w, arena.allocator(), &request, name, sequence, client) catch |err| {
        w.display.print("HTTP request failed: {s}\n", .{@errorName(err)});
    };
}

fn json(a: std.mem.Allocator, request: *Request, status: std.http.Status, value: anytype) !void {
    const body = try std.json.Stringify.valueAlloc(a, value, .{});
    try request.respond(body, .{ .status = status, .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}
fn failure(a: std.mem.Allocator, request: *Request, status: std.http.Status, message: []const u8) !void {
    try json(a, request, status, .{ .@"error" = .{ .message = message, .type = "invalid_request_error" } });
}
fn handle(worker: *Worker, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, client: *control.Client) !void {
    var route = request.head.target;
    if (std.mem.indexOfScalar(u8, route, '?')) |at| route = route[0..at];
    route = std.mem.trimEnd(u8, route, "/");
    const response_id = responses.route(route);
    if (response_id) |id| {
        if (request.head.method == .GET) {
            const found = try worker.response_store.getResponse(a, worker.io, id) orelse return failure(a, request, .not_found, "Response not stored");
            return json(a, request, .ok, found);
        }
        if (request.head.method == .DELETE) {
            if (!worker.response_store.delete(worker.io, id)) return failure(a, request, .not_found, "Response not stored");
            return json(a, request, .ok, .{ .id = id, .object = "response", .deleted = true });
        }
        if (id.len != 0) return failure(a, request, .not_found, "Unknown route");
    }
    if (request.head.method == .GET) {
        if (route.len == 0 or std.mem.eql(u8, route, "/health")) return json(a, request, .ok, .{ .status = "ok", .model = model, .warming = worker.warming.load(.acquire), .max_batch_size = worker.batch_streams, .max_batch_rows = worker.batch_rows, .shared_decode = worker.shared_decode, .background_preemptions = worker.preemptions.load(.acquire), .memory = worker.memory_stats.snapshot(), .prompt_cache = worker.prefix_stats.snapshot(), .inference = worker.stats.snapshot() });
        if (std.mem.eql(u8, route, "/v1/models") or std.mem.eql(u8, route, "/models")) return json(a, request, .ok, .{ .object = "list", .data = &.{.{ .id = model, .object = "model", .created = std.Io.Clock.real.now(worker.io).toSeconds(), .owned_by = "tensorfold" }} });
        return failure(a, request, .not_found, "Unknown route");
    }
    const is_chat = response_id != null or std.mem.eql(u8, route, "/v1/chat/completions") or std.mem.eql(u8, route, "/chat/completions");
    if (request.head.method != .POST or (!is_chat and !std.mem.eql(u8, route, "/v1/completions") and !std.mem.eql(u8, route, "/completions"))) return failure(a, request, .not_found, "Unknown route");
    var body_buffer: [8192]u8 = undefined;
    const body_reader = try request.readerExpectContinue(&body_buffer);
    const bytes = body_reader.allocRemaining(a, .limited(32 * 1024 * 1024)) catch {
        client.cancellation().check() catch |err| return requestFailure(a, request, err);
        return failure(a, request, .bad_request, "Invalid or oversized request body");
    };
    const body = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch return failure(a, request, .bad_request, "Invalid JSON");
    var chat_body = body.value;
    const created = std.Io.Clock.real.now(worker.io).toSeconds();
    var response_reply: ?responses.Reply = null;
    defer if (response_reply) |*reply| reply.deinit();
    if (response_id != null) {
        const translated = responses.translate(a, worker.io, body.value, worker.response_store) catch |err| return failure(a, request, .bad_request, @errorName(err));
        chat_body = translated.chat;
        const id = try std.fmt.allocPrint(a, "resp_{d}_{d}", .{ created, sequence });
        response_reply = .{ .a = a, .io = worker.io, .initial = try responses.base(a, translated, id, model, created), .store = if (translated.store) worker.response_store else null, .added = translated.added };
    }
    const options = inference.Options.parseWithDefaults(a, chat_body, worker.defaults) catch |err| return failure(a, request, .bad_request, @errorName(err));
    if (chat_body.object.get("model")) |value| if (value != .string or !std.mem.eql(u8, value.string, model)) return failure(a, request, .not_found, "Unknown model");
    client.cancellation().check() catch |err| return requestFailure(a, request, err);
    var job = Job{ .stats = worker.stats, .a = a, .request = request, .model = model, .sequence = sequence, .created = created, .client = client, .body = chat_body, .is_chat = is_chat, .options = options, .response_reply = if (response_reply) |*reply| reply else null };
    job.background = background.requested(chat_body, is_chat);
    if (job.background) {
        for (0..30) |_| {
            client.cancellation().check() catch |err| return requestFailure(a, request, err);
            try std.Io.sleep(worker.io, .fromMilliseconds(5), .awake);
        }
    }
    worker.stats.enqueue();
    defer worker.stats.finish(job.activated);
    if (!worker.queue.put(worker.io, &job, false)) return failure(a, request, .service_unavailable, "Inference queue is full");
    job.done.waitUncancelable(worker.io);
    if (job.failure) |err| return err;
}

fn requestFailure(a: std.mem.Allocator, request: *Request, err: anyerror) !void {
    return failure(a, request, switch (err) {
        error.RequestTimedOut => .request_timeout,
        error.ServerStopping => .service_unavailable,
        else => .bad_request,
    }, @errorName(err));
}

const Pending = struct {
    job: *Job,
    id: []const u8 = "",
    thinking: bool = false,
    tools: std.json.Value = .null,
    max_calls: ?usize = null,
    markers: reply_text.Markers = .{},
    gate: ?@import("call_gate.zig").Gate = null,
    initial_gate: ?@import("call_gate.zig").Gate = null,
    replay_tokens: std.ArrayList(u32) = .empty,
    image_sources: []const @import("vision.zig").EncodedImage = &.{},
    image_tokens: []const i32 = &.{},
    prepared_image: ?@import("vision.zig").Prepared = null,
    ids: []const i32 = &.{},
    options: inference.Options = .{},
    image: ?@import("vision.zig").Prompt = null,
    generation: ?inference.RequestGeneration = null,
    response: ?std.http.BodyWriter = null,
    stream: ?Stream = null,
    buffer: [8192]u8 = undefined,
    saved_position: usize = 0,
    cached_tokens: usize = 0,
    think_end: ?u32 = null,
    history_len: usize = 0,
    warm_checkpoint: ?usize = null,
    system_len: usize = 0,
    checkpoints: @import("prompt_cache.zig").Checkpoints = .{},
    shared_checkpoints: @import("prompt_cache.zig").Checkpoints = .{},
    disk: ?*@import("snapshot_store.zig").Store = null,
    prefix_reserve: u64 = 0,
    activation_order: u64 = 0,
    decode_order: u64 = 0,
    growth: enum { run, paused, ended } = .run,
    passed: usize = 0,
    admission_failure: ?anyerror = null,
    reservation_horizon: ?u64 = memory_policy.growth_horizon,

    fn start(w: *Worker, session: *inference.Session, job: *Job) !?*Pending {
        if (job.suspended) |pending| {
            job.suspended = null;
            if (pending.image_sources.len > 0 and pending.prepared_image == null) pending.prepared_image = @import("vision.zig").Prepared.init(session.io, session.directory, pending.image_sources, pending.image_tokens) catch |err| {
                defer pending.deinit();
                try pending.reportError(err);
                return null;
            };
            return pending;
        }
        const p = try job.a.create(Pending);
        p.* = .{ .job = job, .disk = w.disk, .reservation_horizon = if (w.batch_streams > 1) memory_policy.growth_horizon else null };
        p.prepare(w, session) catch |err| {
            defer p.deinit();
            try p.reportError(err);
            return null;
        };
        return p;
    }

    fn deinit(p: *Pending) void {
        p.replay_tokens.deinit(mx.allocator);
        if (p.generation) |*generation| generation.deinit();
        if (p.image) |*image| image.deinit();
        if (p.prepared_image) |*image| image.deinit();
        p.job.a.destroy(p);
    }

    fn preempt(p: *Pending) !void {
        if (p.generation) |*generation| {
            const tokens = generation.tokens();
            if (tokens.len > p.replay_tokens.items.len) {
                p.replay_tokens.clearRetainingCapacity();
                try p.replay_tokens.appendSlice(mx.allocator, tokens);
            }
            generation.deinit();
            p.generation = null;
        }
        if (p.image) |*image| image.deinit();
        p.image = null;
        if (p.prepared_image) |*image| image.deinit();
        p.prepared_image = null;
        p.gate = p.initial_gate;
        p.saved_position = 0;
        p.cached_tokens = 0;
        p.prefix_reserve = 0;
        p.growth = .run;
        p.passed = 0;
        p.admission_failure = null;
        if (p.stream) |*stream| stream.replay.restart(stream.accumulated.items);
        if (p.job.activated) {
            if (p.job.warmer == null) p.job.stats.requeue();
            p.job.activated = false;
        }
    }

    fn reportError(p: *Pending, err: anyerror) !void {
        if (p.response) |*response| {
            if (p.job.response_reply) |reply| {
                _ = try reply.fail(.{ .string = @errorName(err) });
                return response.end();
            }
            const body = try std.json.Stringify.valueAlloc(p.job.a, .{ .@"error" = .{ .message = @errorName(err) } }, .{});
            try response.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{body});
            try response.end();
        } else try p.job.reportError(err);
    }

    fn prepare(p: *Pending, w: *Worker, session: *inference.Session) !void {
        const a = p.job.a;
        const body = p.job.body;
        const cancellation = p.job.cancellation();
        try cancellation.check();
        if (p.job.warmer) |warmer| {
            p.history_len = warmer.jobs.stops[warmer.index];
            p.ids = warmer.jobs.probe[0 .. p.history_len + 1];
            p.options = p.job.options;
            if (warmer.index + 1 == warmer.jobs.stops.len) p.warm_checkpoint = p.history_len;
            try session.validate(p.ids, p.options);
            return;
        }
        if (body.object.get("response_format")) |format| if (format != .null) {
            const kind = if (format == .object) format.object.get("type") orelse std.json.Value.null else std.json.Value.null;
            if (kind != .string or !std.mem.eql(u8, kind.string, "text")) return error.StructuredOutputNotImplemented;
        };
        var options = p.job.options;
        if (!w.drafts) options.draft = false;
        var ids: std.ArrayList(i32) = .empty;
        var raw_images = body.object.get("images") orelse .null;
        if (p.job.is_chat) {
            p.tools = try chat.activeTools(a, body);
            if (body.object.get("parallel_tool_calls")) |parallel| if (parallel != .null) {
                if (parallel != .bool) return error.InvalidParallelToolCalls;
                if (!parallel.bool) p.max_calls = 1;
            };
            const rendered = try session.renderChat(a, body, w.thinking, w.effort);
            try ids.appendSlice(a, try chat.encode(a, &session.tokenizer, rendered));
            raw_images = rendered.images;
            if (raw_images != .array or raw_images.array.items.len == 0) {
                p.history_len = try session.chat_template.?.historyLength(a, &session.tokenizer, body, w.thinking, w.effort, ids.items);
                p.system_len = try session.chat_template.?.systemPrefixLength(a, &session.tokenizer, body, w.thinking, w.effort, ids.items);
            }
            p.thinking = rendered.thinking;
            if (chat.requiresCall(body)) {
                const form = (try session.chat_template.?.callForm(a, &session.tokenizer)) orelse return error.UnsupportedRequiredToolCalls;
                var names: std.ArrayList([]const u8) = .empty;
                if (p.tools == .array) for (p.tools.array.items) |spec| try names.append(a, try chat.toolName(spec));
                p.gate = try @import("call_gate.zig").Gate.init(a, &session.tokenizer, ids.items, form, names.items, session.backend == .gemma);
            }
        } else switch (body.object.get("prompt") orelse return error.MissingPrompt) {
            .string => |value| {
                for (try session.tokenizer.encode(a, value)) |id| try ids.append(a, @intCast(id));
            },
            .array => |values| for (values.items) |value| {
                if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(i32)) return error.InvalidPromptToken;
                try ids.append(a, @intCast(value.integer));
            },
            else => return error.InvalidPrompt,
        }
        if (!p.thinking) options.thinking_budget = 0;
        try session.validate(ids.items, options);
        if (raw_images == .array and raw_images.array.items.len > 0 and session.backend != .qwen) return error.UnsupportedModelImages;
        const images = try @import("image_source.zig").loadWithCancellation(a, session.io, raw_images, w.vision_urls, cancellation);
        p.id = try std.fmt.allocPrint(a, "{s}cmpl-{d}-{d}", .{ if (p.job.is_chat) "chat" else "", p.job.created, p.job.sequence });
        p.markers = if (session.backend == .gemma) reply_text.gemma_markers else .{};
        if (p.thinking) {
            const end = try session.tokenizer.encode(a, p.markers.close);
            if (end.len == 1 and std.mem.eql(u8, try session.tokenizer.decode(a, end, false), p.markers.close)) p.think_end = end[0];
        }
        try cancellation.check();
        if (images.len > 0) {
            p.image_sources = images;
            p.image_tokens = ids.items;
            p.prepared_image = try @import("vision.zig").Prepared.init(session.io, session.directory, images, ids.items);
            p.ids = try a.dupe(i32, p.prepared_image.?.tokens.items);
        } else p.ids = ids.items;
        try session.validate(p.ids, options);
        p.options = options;
        p.initial_gate = p.gate;
    }

    fn activate(p: *Pending, session: *inference.Session) !void {
        const a = p.job.a;
        const cancellation = p.job.cancellation();
        const options = p.options;
        try cancellation.check();
        if (p.job.warmer == null) p.job.stats.activate();
        p.job.activated = true;
        if (p.prepared_image) |*prepared| {
            p.image = try prepared.encode(session.io, session.directory, &session.backend.qwen.weights);
            prepared.deinit();
            p.prepared_image = null;
        }
        if (options.stream and p.response == null) {
            p.response = try p.job.request.?.respondStreaming(&p.buffer, .{ .respond_options = .{ .keep_alive = false, .extra_headers = &.{ .{ .name = "content-type", .value = "text/event-stream" }, .{ .name = "cache-control", .value = "no-cache" } } } });
            p.stream = .{ .a = a, .writer = &p.response.?.writer, .transport = p.job.request.?.server.out, .id = p.id, .model = p.job.model, .created = p.job.created, .is_chat = p.job.is_chat, .thinking = p.thinking, .markers = p.markers, .tools = p.tools, .max_calls = p.max_calls, .cancellation = cancellation, .response_reply = p.job.response_reply };
            if (p.job.response_reply) |reply| {
                reply.context = &p.stream.?;
                reply.emit = Stream.responseEvent;
                try reply.start();
            }
            if (p.job.is_chat) try p.stream.?.chatChunk(.{ .role = "assistant", .content = "" }, null);
        }
        p.generation = try inference.RequestGeneration.init(session, mx.allocator, p.ids, options, .{ .tools = p.tools, .context = if (p.stream) |*stream| stream else null, .emit = if (p.stream != null) Stream.emit else null, .cancellation = cancellation, .gate = if (p.gate) |*gate| gate else null, .replay_tokens = p.replay_tokens.items }, if (p.image) |*image| image else null);
    }

    fn reservedReply(p: *const Pending) u64 {
        return memory_policy.reserveReply(p.options.max_tokens, p.reservation_horizon);
    }

    fn reserve(p: *const Pending, stream: memory_policy.Live) memory_policy.Live {
        return memory_policy.reserveLive(stream, p.ids.len, p.reservation_horizon);
    }

    fn isLivePeer(p: *const Pending, other: *const Pending) bool {
        return other != p and other.options.max_tokens > 0 and other.generation != null;
    }

    fn requestBudget(p: *const Pending, admission: memory_policy.Admission, active: []const ?*Pending) u64 {
        for (active) |slot| if (slot) |other| if (p.isLivePeer(other)) return admission.budget;
        return admission.solo_budget orelse admission.budget;
    }

    const PrefixReservation = struct {
        prefix: memory_policy.Admission.Prefix,
        budget: u64,
    };

    fn reservePrefix(p: *Pending, session: *inference.Session, admission: *memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !?PrefixReservation {
        var live: [shared_round.max_streams]memory_policy.Live = undefined;
        var count: usize = 0;
        var copies: u64 = 0;
        var active_caches: u64 = 0;
        var work_prompt = p.ids.len;
        const budget = p.requestBudget(admission.*, active);
        for (active) |slot| if (slot) |other| if (p.isLivePeer(other)) {
            const generation = &other.generation.?;
            live[count] = other.reserve(generation.memoryLengths());
            count += 1;
            copies +|= other.prefix_reserve;
            copies +|= try admission.initialBytes(live[count - 1]);
            active_caches +|= generation.cacheBytes();
            if (!generation.isDecoding()) work_prompt = @max(work_prompt, other.ids.len);
        };
        const workspace = if (p.prepared_image) |*prepared| prepared.workspaceBytes() else 0;
        const longest = p.ids.len + p.reservedReply();
        var chunks: ?@import("prefill_plan.zig").Chunks = null;
        defer if (chunks) |plan| plan.deinit(mx.allocator);
        var keep: ?[]const i32 = null;
        var prefix_bytes: u64 = 0;
        if (p.prepared_image == null and p.options.max_tokens > 0) if (prefixes) |store| {
            chunks = try (try session.prefillPlan()).chunks(mx.allocator, p.ids);
            if (p.disk) |disk| {
                const extra = (try admission.projected(0, work_prompt, longest, live[0..count])) +| copies +| workspace;
                disk.readBest(store, @as(std.meta.Tag(inference.Backend), session.backend), p.ids, .{ .starts = chunks.?.starts }, extra) catch |err| @import("snapshot_store.zig").report("read", err);
            }
            if (store.best(p.ids, .{ .starts = chunks.?.starts })) |index| {
                keep = store.entries.items[index].tokens;
                prefix_bytes = store.entries.items[index].nbytes;
            }
        };
        var reserved: memory_policy.Admission.Prefix = undefined;
        while (true) {
            const used = try memory_runtime.activeBytes();
            const stored_others = if (prefixes) |store| store.nbytes() -| prefix_bytes else 0;
            reserved = try admission.prefixProjected(used, work_prompt, longest, live[0..count], prefix_bytes, stored_others +| active_caches, copies +| workspace);
            if ((p.options.max_tokens == 0 and workspace == 0) or reserved.take <= budget) break;
            const store = prefixes orelse break;
            const reclaimed = try admission.prefixProjected(used -| stored_others, work_prompt, longest, live[0..count], prefix_bytes, active_caches, copies +| workspace);
            // The selected entry survives reclamation; arrays held elsewhere are remeasured after each eviction.
            if (reclaimed.take > budget or !store.evictOne(keep)) break;
            try mx.check(mx.c.mlx_clear_cache());
        }
        if ((p.options.max_tokens > 0 or workspace > 0) and reserved.take > budget) {
            admission.refused +|= 1;
            if (count == 0) return error.RequestExceedsMemoryBudget;
            return null;
        }
        return .{ .prefix = reserved, .budget = budget };
    }

    fn restorePrefix(p: *Pending, store: *PrefixStore, reservation: PrefixReservation) !void {
        const reserved = reservation.prefix;
        const policy = @import("prompt_cache.zig");
        p.checkpoints = policy.checkpoints(p.history_len, 0, null, p.ids).aligned(p.generation.?.boundary(), 0, p.ids.len);
        p.shared_checkpoints = policy.sharedCheckpoints(p.system_len, p.generation.?.boundary(), p.ids.len);
        if (p.warm_checkpoint) |at| p.shared_checkpoints = .{ .values = .{ at, 0, 0 }, .count = 1 };
        const take = reserved.copy > reservation.budget;
        var hit = try store.match(p.ids, p.generation.?.boundary(), take);
        if (take and hit == null) return error.MissingReservedPrefix;
        if (hit) |*value| {
            defer value.deinit(mx.allocator);
            const size = value.cache.nbytes();
            try p.generation.?.restoreOwnedPrefix(&value.cache);
            p.saved_position = value.count;
            p.cached_tokens = value.count;
            p.prefix_reserve = if (take) reserved.shared else size;
            p.checkpoints = policy.checkpoints(p.history_len, value.count, value.last_prompt, p.ids).aligned(p.generation.?.boundary(), value.count, p.ids.len);
        }
    }

    fn admit(p: *Pending, session: *inference.Session, admission: *memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !bool {
        try p.job.cancellation().check();
        if (p.generation == null) {
            const reserved = (try p.reservePrefix(session, admission, active, prefixes)) orelse return false;
            try mx.check(mx.c.mlx_clear_cache());
            try p.activate(session);
            if (p.image == null and p.options.max_tokens > 0) if (prefixes) |store| try p.restorePrefix(store, reserved);
            if (p.warm_checkpoint) |at| if (p.cached_tokens == at) if (prefixes) |store| {
                p.saved_position = 0;
                try p.savePrefix(store, admission.*, active);
            };
        }
        return true;
    }

    fn advance(p: *Pending, session: *inference.Session, admission: *memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !bool {
        try p.job.cancellation().check();
        if (p.admission_failure) |admission_error| return admission_error;
        if (p.growth == .ended) return error.StreamGrowthExceedsMemoryBudget;
        if (p.growth == .paused) return false;
        if (!try p.admit(session, admission, active, prefixes)) return false;
        if (p.options.max_tokens > 0 and !p.generation.?.isDecoding() and !try p.guardPrefill(admission.*, active, prefixes)) return false;
        const done = try p.step(session);
        if (!done) if (prefixes) |store| {
            const at = p.generation.?.memoryLengths().now;
            if (!p.job.is_chat or p.checkpoints.contains(at) or p.shared_checkpoints.contains(at)) p.savePrefix(store, admission.*, active) catch {};
        };
        return done;
    }

    fn guardPrefill(p: *Pending, admission: memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !bool {
        var others: [shared_round.max_streams]memory_policy.Live = undefined;
        var count: usize = 0;
        var decoding = false;
        var prompt = p.ids.len;
        const budget = p.requestBudget(admission, active);
        for (active) |slot| if (slot) |other| if (p.isLivePeer(other)) {
            if (other.generation) |*generation| {
                others[count] = other.reserve(generation.memoryLengths());
                others[count].copy_bytes = other.prefix_reserve;
                count += 1;
                if (generation.isDecoding()) decoding = true else prompt = @max(prompt, other.ids.len);
            }
        };
        var filling = p.reserve(p.generation.?.memoryLengths());
        filling.copy_bytes = p.prefix_reserve;
        const memory = memory_runtime.Reclaim{ .prefixes = prefixes };
        while (true) {
            const needed = try admission.fillingProjected(try memory.used(), prompt, filling, others[0..count]);
            if (needed <= budget) return true;
            if (needed -| (try memory.freeable()) > budget or !try memory.reclaim()) {
                if (decoding) return false;
                return error.RequestExceedsMemoryBudget;
            }
        }
    }

    fn savePrefix(p: *Pending, store: *PrefixStore, admission: memory_policy.Admission, active: []const ?*Pending) !void {
        const generation = &p.generation.?;
        if (generation.memoryLengths().now <= p.saved_position) return;
        var snapshot = (try generation.snapshot()) orelse return;
        var adopted = false;
        defer if (!adopted) snapshot.deinit();
        const size = snapshot.nbytes();
        if (store.budget_bytes) |budget| if (size > budget and !store.admit_oversize) return;
        var live: [shared_round.max_streams]memory_policy.Live = undefined;
        var count: usize = 0;
        var prompt: usize = 0;
        var copies: u64 = 0;
        for (active) |slot| if (slot) |other| if (other.options.max_tokens > 0) if (other.generation) |*request| {
            live[count] = other.reserve(request.memoryLengths());
            count += 1;
            prompt = @max(prompt, other.ids.len);
            copies +|= other.prefix_reserve;
            copies +|= try admission.initialBytes(live[count - 1]);
        };
        // Each live request can copy the retained buffers independently as it advances.
        const reserved = (copies -| p.prefix_reserve) +| @max(p.prefix_reserve, size);
        var projected = try admission.projected(try memory_runtime.activeBytes(), prompt, 0, live[0..count]);
        while (projected +| reserved > admission.budget) {
            if ((projected +| reserved) -| store.nbytes() > admission.budget or !store.evictOne(null)) return;
            try mx.check(mx.c.mlx_clear_cache());
            projected = try admission.projected(try memory_runtime.activeBytes(), prompt, 0, live[0..count]);
        }
        const position = snapshot.position();
        adopted = true;
        try store.insertOwned(p.ids[0..position], snapshot, p.ids, p.shared_checkpoints.contains(position));
        if (p.shared_checkpoints.contains(position)) if (p.disk) |disk| if (store.entries.items.len > 0) disk.persist(&store.entries.items[0]);
        p.saved_position = position;
        p.prefix_reserve = @max(p.prefix_reserve, size);
    }

    fn step(p: *Pending, session: *inference.Session) !bool {
        const before = p.generation.?.progress();
        const started = live_status.now(session.io);
        const done = try p.generation.?.step(session);
        const ended = live_status.now(session.io);
        const after = p.generation.?.progress();
        p.job.stats.record(after.prefilled - before.prefilled, after.decoded - before.decoded, started, ended);
        p.recordDraftProgress(before, after);
        if (!done) return false;
        return p.finishReply(session);
    }

    fn recordDraftProgress(p: *Pending, before: inference.RequestGeneration.Progress, after: inference.RequestGeneration.Progress) void {
        p.job.stats.recordDrafts(after.proposed - before.proposed, after.accepted - before.accepted, after.structural_proposed - before.structural_proposed, after.structural_accepted - before.structural_accepted);
        p.job.stats.recordNeural(after.neural_proposed - before.neural_proposed, after.neural_accepted - before.neural_accepted);
    }

    fn finishReply(p: *Pending, session: *inference.Session) !bool {
        var reply = try p.generation.?.takeReply();
        defer reply.deinit(mx.allocator);
        if (p.job.warmer != null) return true;
        const a = p.job.a;
        const request = p.job.request.?;
        const id = p.id;
        const model = p.job.model;
        const created = p.job.created;
        const usage = .{ .prompt_tokens = reply.prompt_tokens, .completion_tokens = reply.tokens.items.len, .total_tokens = reply.prompt_tokens + reply.tokens.items.len, .prompt_tokens_details = .{ .cached_tokens = p.cached_tokens }, .completion_tokens_details = .{ .reasoning_tokens = reply_text.reasoningCount(reply.tokens.items, p.think_end) } };
        if (p.response) |*response| {
            const state = &p.stream.?;
            if (p.job.is_chat) try state.chatText(true);
            try state.finishWithUsage(if (state.calls_sent > 0) "tool_calls" else @tagName(reply.finish_reason), usage);
            if (p.job.response_reply == null) try response.writer.writeAll("data: [DONE]\n\n");
            try response.end();
        } else {
            if (p.job.is_chat) {
                const parts = if (p.thinking) reply_text.splitThinking(reply.content, true, p.markers) else reply_text.Parts{ .content = reply.content };
                var parsed = try tool_calls.parse(a, parts.content, p.tools, p.max_calls, id);
                if (p.max_calls != null and p.tools == .array and p.tools.array.items.len > 0) parsed.content = try tool_calls.singleContent(a, parsed.content);
                const completion = .{ .id = id, .object = "chat.completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .message = .{ .role = "assistant", .content = parsed.content, .reasoning_content = parts.reasoning, .tool_calls = parsed.calls }, .finish_reason = if (parsed.calls.len > 0) "tool_calls" else @tagName(reply.finish_reason) }}, .usage = usage };
                if (p.job.response_reply) |result| {
                    try json(a, request, .ok, try result.complete(completion, std.Io.Clock.real.now(session.io).toSeconds()));
                } else try json(a, request, .ok, completion);
            } else try json(a, request, .ok, .{ .id = id, .object = "text_completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .text = reply.content, .finish_reason = @tagName(reply.finish_reason), .logprobs = @as(?u8, null) }}, .usage = usage });
        }
        return true;
    }
};

const Stream = struct {
    response_reply: ?*responses.Reply = null,
    a: std.mem.Allocator,
    writer: *std.Io.Writer,
    transport: *std.Io.Writer,
    id: []const u8,
    model: []const u8,
    created: i64,
    cancellation: @import("cancellation.zig").Cancellation,
    is_chat: bool = false,
    thinking: bool = false,
    markers: reply_text.Markers = .{},
    accumulated: std.ArrayList(u8) = .empty,
    replay: background.Replay = .{},
    sent_content: usize = 0,
    sent_reasoning: usize = 0,
    tools: std.json.Value = .null,
    max_calls: ?usize = null,
    calls_sent: usize = 0,
    tool_stream: @import("tool_stream.zig").Streamer = .{},
    fn responseEvent(context: ?*anyopaque, event: std.json.Value) !void {
        const s: *Stream = @ptrCast(@alignCast(context.?));
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const body = try std.json.Stringify.valueAlloc(scratch.allocator(), event, .{});
        try s.writer.print("event: {s}\ndata: {s}\n\n", .{ event.object.get("type").?.string, body });
        try s.writer.flush();
        try s.transport.flush();
    }
    fn finishWithUsage(s: *Stream, reason: []const u8, usage: anytype) !void {
        if (s.response_reply) |reply| {
            _ = try reply.finishUsage(reason, usage, std.Io.Clock.real.now(reply.io).toSeconds());
            return;
        }
        const body = if (s.is_chat)
            try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "chat.completion.chunk", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .delta = std.json.Value{ .object = .empty }, .finish_reason = reason }}, .usage = usage }, .{})
        else
            try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "text_completion", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .text = "", .finish_reason = reason, .logprobs = @as(?u8, null) }}, .usage = usage }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn chunk(s: *Stream, value: []const u8, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "text_completion", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .text = value, .finish_reason = finish, .logprobs = @as(?u8, null) }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn emit(context: ?*anyopaque, value: []const u8) !void {
        if (value.len == 0) return;
        const s: *Stream = @ptrCast(@alignCast(context.?));
        try s.cancellation.check();
        const fresh = try s.replay.fresh(s.accumulated.items, value);
        if (fresh.len == 0) return;
        try s.accumulated.appendSlice(s.a, fresh);
        if (!s.is_chat) return s.chunk(fresh, null);
        try s.chatText(false);
    }
    fn chatChunk(s: *Stream, delta: anytype, finish: ?[]const u8) !void {
        if (s.response_reply) |reply| return reply.writeDelta(delta);
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "chat.completion.chunk", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .delta = delta, .finish_reason = finish }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn chatText(s: *Stream, finished: bool) !void {
        const parts = if (s.thinking) reply_text.splitThinking(s.accumulated.items, finished, s.markers) else reply_text.Parts{ .content = s.accumulated.items };
        if (parts.reasoning.len > s.sent_reasoning) {
            try s.chatChunk(.{ .reasoning_content = parts.reasoning[s.sent_reasoning..] }, null);
            s.sent_reasoning = parts.reasoning.len;
        }
        const has_tools = s.tools == .array and s.tools.array.items.len > 0;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const a = scratch.allocator();
        if (has_tools and s.max_calls == null) try s.tool_stream.feed(a, parts.content, s.tools, s, toolDelta);
        var parsed = if (finished) try tool_calls.parse(a, parts.content, s.tools, s.max_calls, s.id) else tool_calls.Result{ .content = if (has_tools) try tool_calls.preview(a, parts.content, s.tools, s.max_calls) else parts.content };
        if (s.max_calls != null and finished and has_tools) parsed.content = try tool_calls.singleContent(a, parsed.content);
        if (parsed.content.len > s.sent_content) {
            try s.chatChunk(.{ .content = parsed.content[s.sent_content..] }, null);
            s.sent_content = parsed.content.len;
        }
        for (parsed.calls, 0..) |call, index| {
            if (index < s.tool_stream.count) continue;
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .id = call.id, .type = "function", .function = .{ .name = call.function.name, .arguments = "" } }} }, null);
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .function = .{ .arguments = call.function.arguments } }} }, null);
            s.calls_sent += 1;
        }
    }
    fn toolDelta(context: ?*anyopaque, delta: @import("tool_stream.zig").Delta) !void {
        const s: *Stream = @ptrCast(@alignCast(context.?));
        if (delta.name) |name| {
            const id = try std.fmt.allocPrint(s.a, "call_{s}_{d}", .{ s.id, delta.index });
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .id = id, .type = "function", .function = .{ .name = name, .arguments = "" } }} }, null);
            s.calls_sent += 1;
        } else try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .function = .{ .arguments = delta.arguments } }} }, null);
    }
};
