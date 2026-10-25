//! The pipeline census: make every function and pipeline state a captured manifest names, through
//! the runtime, one at a time, and report per state what the translator and the driver did: the
//! translator's wall and CPU time, the SPIR-V words of the largest module the state's first
//! `VkPipeline` is made from, and the driver's per-shader statistics where it prints them (Mesa's
//! ir3, `IR3_SHADER_DEBUG=disasm`: instructions, nops, ALU categories, registers, waves).
//!
//! `pipeline-census run --manifest FILE --out DIR [--only ID[,ID...]] [--unorm8-filter exact]
//! [--keep-logs]`
//!
//! It is the shader leg of `command-replay`: the same manifest (version 1), of which it reads the
//! `library`, `libraries`, `functions`, `renderPipelines` and `computePipelines` tables and no
//! command. The tables take three keys `command-replay` does not read: a function constant's
//! `"name"` (a named constant, in place of `"index"`), a render pipeline's `"vertexDescriptor"`
//! (`{"attributes": [{"index", "format", "offset", "buffer"}], "layouts": [{"index", "stride",
//! "step", "rate"}]}`, raw Metal enums) and `"rasterization"` (default true), and a pipeline's
//! `"label"`. A function without `"library"` is looked up by name in each library in turn. A
//! metallib with several slices is read at its AIR64 slice, as the Apple platforms' Metal reads it.
//!
//! A state's creation makes the `VkPipeline` its first draw is predicted to take, inside the
//! creation, so the driver compiles each state's shaders here. A draw whose bindings key another
//! pipeline makes its own at the draw: the manifest's `draws` table (`{DID: {"render": PID,
//! "primitive", "indexType"?, "pass": {"colors": [{"format", "samples"} or null], "depth"?,
//! "stencil"?, "samples"}, "depthStencil"?, "raster"?: [cull, winding, fill], "slots": {"<v|f><b|t|s><index>":
//! {"buffer": length, "offset"} | {"bytes": length} | <texture: "type", "format", "width",
//! "height", "depth", "levels", "samples", "array", "usage", "options", "over"?, "offset", "row">
//! | <sampler: command-replay's keys and "lodAverage">}}}`) encodes each draw alone, of three
//! vertices, over zeroed resources of those shapes, in a pass of at most 64x64, commits it and
//! waits for the runtime to go quiet; its row reports what the draw made (`on_predicted`: the
//! draw ran on its state's predicted pipeline; `variant_pipelines`). The device
//! is the default device (`VK_DRIVER_FILES` and the ICD's own variables choose it), opened with
//! the newest Metal level, a new empty shader cache, the creation log on
//! (`METAL2VULKAN_LOG_PIPELINES`), and, unless `--unorm8-filter exact`, the device's own 8-bit
//! linear filter where its facts allow (`Unorm8LinearFilterPolicy::DeviceWithinOneCode`), as an
//! embedder that takes the device's measured filter opens it.
//!
//! Each creation runs with the process's standard output and error sent to a file, which is then
//! parsed for the driver's statistics lines (`; <STAGE> prog <n>/<v>: ...`). `--keep-logs` keeps
//! each state's file as `DIR/logs/<id>.log`.
//!
//! **Outputs.** `DIR/census.tsv`: one row per state or draw and driver shader (a state whose driver printed
//! nothing has one row with empty statistics), columns named on its first line. `DIR/census.jsonl`:
//! one object per state with the same values. `DIR/status.txt`: the device, each failure, and `ok`.
//! The exit status is 0 when every state was made, 1 when any was refused, 2 for a harness failure.
//!
//! GUEST DATA: a manifest made from a guest's capture, its libraries and every output here are the
//! consumer's: they stay on the hosts the guest data rule allows and are never committed.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, CullMode, DataType, IndexType,
    PixelFormat, PrimitiveTopologyClass, PrimitiveType, SamplerAddressMode, SamplerBorderColor,
    SamplerMinMagFilter, SamplerMipFilter, StencilOperation, TextureType, TextureUsage,
    TriangleFillMode, VertexFormat, VertexStepFunction, Winding,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions, Function, Library,
    LoadAction, MessageSink, MetalLevel, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, ShaderCacheOptions, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan_validation::candidate::runtime_case::{open_device, ScratchCache};
use metal2vulkan_validation::gpu::DeviceType;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const USAGE: &str = "usage: pipeline-census run --manifest FILE --out DIR [--only ID[,ID...]] \
                     [--unorm8-filter exact] [--keep-logs]";

/// The statistics columns of one driver shader, in the order the driver prints them.
const STATS: [&str; 26] = [
    "instr",
    "nops",
    "non_nops",
    "mov",
    "cov",
    "dwords",
    "last_baryf",
    "last_helper",
    "half",
    "full",
    "constlen",
    "cat0",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "sstall",
    "ss",
    "systall",
    "sy",
    "loops",
    "max_waves",
    "double_threadsize",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") => match options(&args[1..]) {
            Ok(options) => run(&options),
            Err(why) => {
                eprintln!("pipeline-census: {why}\n{USAGE}");
                2
            }
        },
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}

fn options(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let name = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected {flag:?}"))?;
        if name == "keep-logs" {
            out.insert(name.to_string(), String::new());
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        out.insert(name.to_string(), value.clone());
    }
    for name in ["manifest", "out"] {
        if !out.contains_key(name) {
            return Err(format!("--{name} is required"));
        }
    }
    Ok(out)
}

/// A metallib's AIR64 slice, or the bytes themselves where they are not a fat file.
fn air_slice(bytes: &[u8]) -> Option<&[u8]> {
    const AIR64: u32 = 0x0100_0017;
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|w| u32::from_be_bytes([w[0], w[1], w[2], w[3]]))
    };
    if word(0)? != 0xcafe_babe {
        return Some(bytes);
    }
    let count = word(4)? as usize;
    for i in 0..count.min(16) {
        let at = 8 + 20 * i;
        if word(at)? == AIR64 {
            let (offset, size) = (word(at + 8)? as usize, word(at + 12)? as usize);
            return bytes.get(offset..offset.checked_add(size)?);
        }
    }
    None
}

fn table(v: &Value, key: &str) -> Vec<(String, Value)> {
    v.get(key)
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn u(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn b(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn s<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key:?} is required"))
}

fn raw<T: TryFrom<u32>>(v: &Value, key: &str, default: u32) -> Result<T, String>
where
    T::Error: std::fmt::Debug,
{
    let n = u32::try_from(u(v, key, u64::from(default))).map_err(|e| e.to_string())?;
    T::try_from(n).map_err(|e| format!("{key} {n}: {e:?}"))
}

fn hex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err(format!("odd hex {text:?}"));
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

/// The calling thread's CPU time.
fn thread_cpu() -> std::time::Duration {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid timespec out-parameter.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    std::time::Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

/// The process's standard output and error sent to a file while it lives.
struct Capture {
    saved: [i32; 2],
    file: std::fs::File,
}

impl Capture {
    fn begin(path: &Path) -> std::io::Result<Capture> {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        let file = std::fs::File::create(path)?;
        // SAFETY: duplicating and replacing the process's own standard descriptors.
        let saved = unsafe { [libc::dup(1), libc::dup(2)] };
        if saved.contains(&-1) {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: as above; `file` stays open for the capture's life.
        unsafe {
            libc::dup2(file.as_raw_fd(), 1);
            libc::dup2(file.as_raw_fd(), 2);
        }
        Ok(Capture { saved, file })
    }

    fn end(self) -> std::io::Result<()> {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: restoring the descriptors `begin` saved.
        unsafe {
            libc::dup2(self.saved[0], 1);
            libc::dup2(self.saved[1], 2);
            libc::close(self.saved[0]);
            libc::close(self.saved[1]);
        }
        self.file.sync_all()
    }
}

/// One driver shader's statistics: its stage name and program, and the numbers of [`STATS`].
struct Shader {
    stage: String,
    program: String,
    stats: BTreeMap<&'static str, u64>,
}

/// The driver statistics lines of a creation's output: `; <STAGE> prog <n>/<v>: <k> <name>, ...`,
/// grouped by stage and program in the order they first appear.
fn shaders(text: &str) -> Vec<Shader> {
    let mut out: Vec<Shader> = Vec::new();
    for line in text.lines() {
        let Some(at) = line.find("; ") else { continue };
        let rest = &line[at + 2..];
        let mut words = rest.splitn(3, ' ');
        let (Some(stage), Some("prog"), Some(tail)) = (words.next(), words.next(), words.next())
        else {
            continue;
        };
        let Some((program, values)) = tail.split_once(": ") else {
            continue;
        };
        let index = match out
            .iter()
            .position(|s| s.stage == stage && s.program == program)
        {
            Some(i) => i,
            None => {
                out.push(Shader {
                    stage: stage.to_string(),
                    program: program.to_string(),
                    stats: BTreeMap::new(),
                });
                out.len() - 1
            }
        };
        for item in values.split(", ") {
            let Some((n, name)) = item.trim().split_once(' ') else {
                continue;
            };
            let key = name.replace(['-', ' '], "_");
            let key = match key.as_str() {
                "instr" | "instrs" => "instr",
                other => match STATS.iter().find(|k| **k == other) {
                    Some(k) => k,
                    None => continue,
                },
            };
            if let Ok(n) = n.parse() {
                out[index].stats.insert(key, n);
            }
        }
    }
    out
}

/// What the creation log said about the pipelines a state's creation made.
fn largest_words(lines: &[String]) -> Option<u64> {
    lines
        .iter()
        .filter_map(|l| {
            let l = l.strip_prefix("metal2vulkan-pipeline-creation: ")?;
            let at = l.find("largest module ")? + "largest module ".len();
            l[at..].split(' ').next()?.parse::<u64>().ok()
        })
        .max()
}

/// One state's or draw's rows: one per driver shader (one with empty statistics where the driver
/// printed none) in the TSV, and its record with the shaders in the JSON lines.
fn write_rows(
    tsv: &mut std::fs::File,
    jsonl: &mut std::fs::File,
    base: &[String],
    found: &[Shader],
    mut record: Value,
) -> Result<(), String> {
    let rows: Vec<Option<&Shader>> = match found.is_empty() {
        true => vec![None],
        false => found.iter().map(Some).collect(),
    };
    for shader in rows {
        let mut cells: Vec<String> = base.to_vec();
        cells.push(shader.map(|s| s.stage.clone()).unwrap_or_default());
        cells.push(shader.map(|s| s.program.clone()).unwrap_or_default());
        for k in STATS {
            cells.push(
                shader
                    .and_then(|s| s.stats.get(k))
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
            );
        }
        writeln!(tsv, "{}", cells.join("\t")).map_err(|e| e.to_string())?;
    }
    record["shaders"] = found
        .iter()
        .map(|s| json!({"stage": s.stage, "program": s.program, "stats": s.stats}))
        .collect();
    writeln!(jsonl, "{record}").map_err(|e| e.to_string())
}

struct Census {
    device: Device,
    libraries: Vec<(String, Library, Vec<String>)>,
    functions: BTreeMap<String, Function>,
    function_us: BTreeMap<String, (u128, u128, String)>,
    creation_log: Arc<Mutex<Vec<String>>>,
    out: PathBuf,
    keep_logs: bool,
    states: BTreeMap<String, RenderPipelineState>,
    queue: Option<CommandQueue>,
}

fn run(options: &BTreeMap<String, String>) -> i32 {
    let manifest_path = PathBuf::from(&options["manifest"]);
    let out = PathBuf::from(&options["out"]);
    if let Err(error) = std::fs::create_dir_all(out.join("logs")) {
        eprintln!("pipeline-census: {}: {error}", out.display());
        return 2;
    }
    let manifest: Value = match std::fs::read(&manifest_path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(why) => {
            eprintln!("pipeline-census: manifest: {why}");
            return 2;
        }
    };
    let dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mut status = match std::fs::File::create(out.join("status.txt")) {
        Ok(f) => f,
        Err(error) => {
            eprintln!("pipeline-census: status.txt: {error}");
            return 2;
        }
    };
    let cache = match ScratchCache::new() {
        Ok(c) => c,
        Err(error) => {
            eprintln!("pipeline-census: shader cache: {error}");
            return 2;
        }
    };
    std::env::set_var("METAL2VULKAN_LOG_PIPELINES", "1");
    let creation_log: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = Arc::clone(&creation_log);
    let mut device_options = DeviceOptions::default();
    device_options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    device_options.metal_level = Some(MetalLevel::NEWEST);
    if options.get("unorm8-filter").map(String::as_str) != Some("exact") {
        device_options.policy.unorm8_linear_filter =
            metal2vulkan::device::Unorm8LinearFilterPolicy::DeviceWithinOneCode;
    }
    device_options.messages = Some(MessageSink::new(move |m| {
        if let Ok(mut lines) = sink.lock() {
            lines.push(format!("{}: {}", m.id_name, m.text));
        }
    }));
    let device = match open_device(DeviceType::Gpu, device_options) {
        Ok(d) => d,
        Err(error) => {
            let _ = writeln!(status, "error device: {error}");
            eprintln!("pipeline-census: device: {error}");
            return 2;
        }
    };
    let _ = writeln!(status, "device {}", device.name());
    let mut census = Census {
        device,
        libraries: Vec::new(),
        functions: BTreeMap::new(),
        function_us: BTreeMap::new(),
        creation_log,
        out: out.clone(),
        keep_logs: options.contains_key("keep-logs"),
        states: BTreeMap::new(),
        queue: None,
    };
    let only: Option<Vec<String>> = options
        .get("only")
        .map(|o| o.split(',').map(str::to_string).collect());
    match census.all(&manifest, &dir, only.as_deref(), &mut status) {
        Ok(refused) => {
            let _ = writeln!(status, "ok {refused} refused");
            eprintln!("pipeline-census: {}", out.join("census.tsv").display());
            i32::from(refused > 0)
        }
        Err(why) => {
            let _ = writeln!(status, "error {why}");
            eprintln!("pipeline-census: {why}");
            2
        }
    }
}

impl Census {
    fn all(
        &mut self,
        m: &Value,
        dir: &Path,
        only: Option<&[String]>,
        status: &mut std::fs::File,
    ) -> Result<usize, String> {
        let named = m
            .get("library")
            .and_then(Value::as_str)
            .map(|file| (String::new(), Value::from(file)));
        for (id, file) in named.into_iter().chain(table(m, "libraries")) {
            let at = format!("library {id:?}");
            let file = file
                .as_str()
                .ok_or_else(|| format!("{at}: not a file name"))?;
            let bytes = std::fs::read(dir.join(file)).map_err(|e| format!("{at}: {e}"))?;
            let air = air_slice(&bytes).ok_or_else(|| format!("{at}: no AIR64 slice"))?;
            let library = self
                .device
                .new_library_with_data(air)
                .map_err(|e| format!("{at}: {e}"))?;
            let names = library.function_names();
            self.libraries.push((id, library, names));
        }
        let mut tsv =
            std::fs::File::create(self.out.join("census.tsv")).map_err(|e| e.to_string())?;
        let mut jsonl =
            std::fs::File::create(self.out.join("census.jsonl")).map_err(|e| e.to_string())?;
        let head = [
            "id",
            "kind",
            "label",
            "ok",
            "vertex",
            "fragment",
            "fn_wall_us",
            "fn_cpu_us",
            "wall_us",
            "cpu_us",
            "spirv_words",
            "stage",
            "program",
        ];
        let columns: Vec<&str> = head.iter().copied().chain(STATS).collect();
        writeln!(tsv, "{}", columns.join("\t")).map_err(|e| e.to_string())?;
        let functions = table(m, "functions");
        let mut refused = 0;
        let mut states: Vec<(String, &str, Value)> = table(m, "renderPipelines")
            .into_iter()
            .map(|(id, v)| (id, "render", v))
            .collect();
        states.extend(
            table(m, "computePipelines")
                .into_iter()
                .map(|(id, v)| (id, "compute", v)),
        );
        for (id, kind, pv) in states {
            if only.is_some_and(|o| !o.contains(&id)) {
                continue;
            }
            let stage_ids: Vec<String> = match kind {
                "render" => ["vertex", "fragment"]
                    .iter()
                    .filter_map(|k| pv.get(*k).and_then(Value::as_str).map(str::to_string))
                    .collect(),
                _ => vec![s(&pv, "function")?.to_string()],
            };
            let log_path = self.out.join("logs").join(format!("{id}.log"));
            let capture = Capture::begin(&log_path).map_err(|e| format!("capture: {e}"))?;
            if let Ok(mut lines) = self.creation_log.lock() {
                lines.clear();
            }
            let mut result = Ok(());
            for fid in &stage_ids {
                if let Err(why) = self.function(fid, &functions) {
                    result = Err(why);
                    break;
                }
            }
            let (wall0, cpu0) = (Instant::now(), thread_cpu());
            if result.is_ok() {
                result = match kind {
                    "render" => self.render(&pv).map(|state| {
                        self.states.insert(id.clone(), state);
                    }),
                    _ => self.compute(&pv),
                };
            }
            let (wall, cpu) = (wall0.elapsed(), thread_cpu() - cpu0);
            capture.end().map_err(|e| format!("capture: {e}"))?;
            let text = std::fs::read_to_string(&log_path).unwrap_or_default();
            self.finish_log(&log_path);
            let words = self
                .creation_log
                .lock()
                .ok()
                .and_then(|l| largest_words(&l));
            let found = shaders(&text);
            let (fn_wall, fn_cpu) = stage_ids
                .iter()
                .filter_map(|f| self.function_us.get(f))
                .fold((0, 0), |(w, c), (fw, fc, _)| (w + fw, c + fc));
            let name_of = |key: &str| -> String {
                pv.get(key)
                    .and_then(Value::as_str)
                    .and_then(|f| self.function_us.get(f))
                    .map(|(_, _, n)| n.clone())
                    .unwrap_or_default()
            };
            let label = pv.get("label").and_then(Value::as_str).unwrap_or("-");
            let (vertex, fragment) = match kind {
                "render" => (name_of("vertex"), name_of("fragment")),
                _ => (name_of("function"), String::new()),
            };
            if let Err(why) = &result {
                refused += 1;
                let _ = writeln!(status, "refused {id}: {why}");
            }
            let base = [
                id.clone(),
                kind.to_string(),
                label.to_string(),
                u8::from(result.is_ok()).to_string(),
                vertex.clone(),
                fragment.clone(),
                fn_wall.to_string(),
                fn_cpu.to_string(),
                wall.as_micros().to_string(),
                cpu.as_micros().to_string(),
                words.map(|w| w.to_string()).unwrap_or_default(),
            ];
            let record = json!({
                "id": id, "kind": kind, "label": label, "ok": result.is_ok(),
                "error": result.as_ref().err(), "vertex": vertex, "fragment": fragment,
                "fn_wall_us": fn_wall, "fn_cpu_us": fn_cpu,
                "wall_us": wall.as_micros() as u64, "cpu_us": cpu.as_micros() as u64,
                "spirv_words": words,
            });
            write_rows(&mut tsv, &mut jsonl, &base, &found, record)?;
        }
        refused += self.draws(m, only, status, &mut tsv, &mut jsonl)?;
        Ok(refused)
    }

    /// Encodes each of the manifest's `draws` on its state, alone in a pass of its own, and commits
    /// it: the `VkPipeline` the draw's bindings key is made at the draw. The number refused.
    fn draws(
        &mut self,
        m: &Value,
        only: Option<&[String]>,
        status: &mut std::fs::File,
        tsv: &mut std::fs::File,
        jsonl: &mut std::fs::File,
    ) -> Result<usize, String> {
        let mut refused = 0;
        for (id, dv) in table(m, "draws") {
            let Ok(pid) = s(&dv, "render") else { continue };
            if only.is_some_and(|o| !o.iter().any(|x| x == &id || x == pid)) {
                continue;
            }
            if !self.states.contains_key(pid) {
                continue;
            }
            if self.queue.is_none() {
                self.queue = Some(self.device.new_command_queue().map_err(|e| e.to_string())?);
            }
            let log_path = self.out.join("logs").join(format!("{id}.log"));
            let capture = Capture::begin(&log_path).map_err(|e| format!("capture: {e}"))?;
            if let Ok(mut lines) = self.creation_log.lock() {
                lines.clear();
            }
            let (wall0, cpu0) = (Instant::now(), thread_cpu());
            let before = self.device.work_counts();
            let result = self.draw(pid, &dv);
            self.settle();
            let after = self.device.work_counts();
            let (wall, cpu) = (wall0.elapsed(), thread_cpu() - cpu0);
            capture.end().map_err(|e| format!("capture: {e}"))?;
            let text = std::fs::read_to_string(&log_path).unwrap_or_default();
            self.finish_log(&log_path);
            let words = self
                .creation_log
                .lock()
                .ok()
                .and_then(|l| largest_words(&l));
            let found = shaders(&text);
            if let Err(why) = &result {
                refused += 1;
                let _ = writeln!(status, "refused draw {id}: {why}");
            }
            let p = table(m, "renderPipelines")
                .into_iter()
                .find(|(k, _)| k == pid)
                .map(|(_, v)| v)
                .unwrap_or_default();
            let name_of = |key: &str| -> String {
                p.get(key)
                    .and_then(Value::as_str)
                    .and_then(|f| self.function_us.get(f))
                    .map(|(_, _, n)| n.clone())
                    .unwrap_or_default()
            };
            let label = p.get("label").and_then(Value::as_str).unwrap_or("-");
            let (vertex, fragment) = (name_of("vertex"), name_of("fragment"));
            let base = [
                id.clone(),
                format!("draw:{pid}"),
                label.to_string(),
                u8::from(result.is_ok()).to_string(),
                vertex.clone(),
                fragment.clone(),
                String::new(),
                String::new(),
                wall.as_micros().to_string(),
                cpu.as_micros().to_string(),
                words.map(|w| w.to_string()).unwrap_or_default(),
            ];
            let record = json!({
                "id": id, "kind": "draw", "render": pid, "label": label, "ok": result.is_ok(),
                "on_predicted": after.creation_pipelines_drawn - before.creation_pipelines_drawn,
                "variant_pipelines": after.variant_pipelines_made - before.variant_pipelines_made,
                "shader_modules": after.shader_modules_made - before.shader_modules_made,
                "error": result.as_ref().err(), "vertex": vertex, "fragment": fragment,
                "wall_us": wall.as_micros() as u64, "cpu_us": cpu.as_micros() as u64,
                "spirv_words": words,
            });
            write_rows(tsv, jsonl, &base, &found, record)?;
        }
        Ok(refused)
    }

    /// One draw of state `pid` with the bindings `dv` records, committed and waited for.
    fn draw(&self, pid: &str, dv: &Value) -> Result<(), String> {
        let state = &self.states[pid];
        let queue = self.queue.as_ref().ok_or("no queue")?;
        let pass_v = dv.get("pass").cloned().unwrap_or_default();
        let size = |key: &str| u(&pass_v, key, 16).clamp(1, 64);
        let (width, height) = (size("width"), size("height"));
        let attachment = |a: &Value| -> Result<Option<Texture>, String> {
            if a.is_null() || a.get("format").is_none() {
                return Ok(None);
            }
            let format: PixelFormat = raw(a, "format", 0)?;
            let mut d = TextureDescriptor::texture_2d(format, width, height, false);
            d.sample_count = u(a, "samples", 1).max(1);
            if d.sample_count > 1 {
                d.texture_type = TextureType::Type2DMultisample;
            }
            d.usage = TextureUsage::RENDER_TARGET;
            self.device
                .new_texture(&d)
                .map(Some)
                .map_err(|e| format!("attachment: {e}"))
        };
        let mut pass = RenderPassDescriptor::new();
        for (k, c) in pass_v
            .get("colors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if k < pass.color_attachments.len() {
                if let Some(t) = attachment(c)? {
                    let a = &mut pass.color_attachments[k];
                    a.texture = Some(t);
                    a.load_action = LoadAction::Clear;
                    a.store_action = StoreAction::Store;
                }
            }
        }
        let depth = pass_v.get("depth").cloned().unwrap_or_default();
        let stencil = pass_v.get("stencil").cloned().unwrap_or_default();
        let depth_texture = attachment(&depth)?;
        let stencil_texture = match (stencil.get("format"), depth.get("format")) {
            (Some(a), Some(b)) if a == b => depth_texture.clone(),
            _ => attachment(&stencil)?,
        };
        if let Some(t) = depth_texture {
            pass.depth_attachment.texture = Some(t);
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::DontCare;
        }
        if let Some(t) = stencil_texture {
            pass.stencil_attachment.texture = Some(t);
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.store_action = StoreAction::DontCare;
        }
        pass.render_target_width = width;
        pass.render_target_height = height;
        pass.default_raster_sample_count = u(&pass_v, "samples", 1).max(1);
        let cb = queue.command_buffer().map_err(|e| e.to_string())?;
        let mut enc = cb
            .render_command_encoder(&pass)
            .map_err(|e| format!("render encoder: {e}"))?;
        enc.set_render_pipeline_state(state)
            .map_err(|e| format!("pipeline: {e}"))?;
        // Every resource lives until the command buffer completed.
        let mut keep: (Vec<Buffer>, Vec<Texture>, Vec<SamplerState>) = Default::default();
        let slots = dv
            .get("slots")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (slot, bv) in &slots {
            let (stage, rest) = slot.split_at(1);
            let (kind, index) = rest.split_at(1);
            let index: u32 = index.parse().map_err(|_| format!("slot {slot}"))?;
            let fragment = match stage {
                "v" => false,
                "f" => true,
                _ => continue,
            };
            let at = format!("slot {slot}");
            match kind {
                "b" if bv.get("bytes").is_some() => {
                    let bytes = vec![0u8; u(bv, "bytes", 0) as usize];
                    match fragment {
                        true => enc.set_fragment_bytes(&bytes, index),
                        false => enc.set_vertex_bytes(&bytes, index),
                    }
                    .map_err(|e| format!("{at}: {e}"))?;
                }
                "b" => {
                    let offset = u(bv, "offset", 0);
                    let length = u(bv, "buffer", 4096).clamp(offset + 16, 1 << 22);
                    let buffer = self
                        .device
                        .new_buffer(length, ResourceOptions::shared())
                        .map_err(|e| format!("{at}: {e}"))?;
                    match fragment {
                        true => enc.set_fragment_buffer(Some(&buffer), offset, index),
                        false => enc.set_vertex_buffer(Some(&buffer), offset, index),
                    }
                    .map_err(|e| format!("{at}: {e}"))?;
                    keep.0.push(buffer);
                }
                "t" => {
                    let texture = self
                        .texture(bv, &mut keep.0)
                        .map_err(|e| format!("{at}: {e}"))?;
                    match fragment {
                        true => enc.set_fragment_texture(Some(&texture), index),
                        false => enc.set_vertex_texture(Some(&texture), index),
                    }
                    .map_err(|e| format!("{at}: {e}"))?;
                    keep.1.push(texture);
                }
                "s" => {
                    let sampler = self.sampler(bv).map_err(|e| format!("{at}: {e}"))?;
                    match fragment {
                        true => enc.set_fragment_sampler_state(Some(&sampler), index),
                        false => enc.set_vertex_sampler_state(Some(&sampler), index),
                    }
                    .map_err(|e| format!("{at}: {e}"))?;
                    keep.2.push(sampler);
                }
                _ => return Err(format!("{at}: unknown kind")),
            }
        }
        let depth_stencil = match dv.get("depthStencil").filter(|v| v.is_object()) {
            Some(ds) => {
                let mut d = DepthStencilDescriptor::default();
                d.depth_compare_function = raw::<CompareFunction>(ds, "compare", 7)?;
                d.depth_write_enabled = b(ds, "write", false);
                if let Some(sv) = ds.get("stencil").filter(|v| v.is_object()) {
                    let mut face = StencilDescriptor::default();
                    face.stencil_compare_function = raw::<CompareFunction>(sv, "compare", 7)?;
                    face.stencil_failure_operation = raw::<StencilOperation>(sv, "fail", 0)?;
                    face.depth_failure_operation = raw::<StencilOperation>(sv, "depthFail", 0)?;
                    face.depth_stencil_pass_operation = raw::<StencilOperation>(sv, "pass", 0)?;
                    face.read_mask = u(sv, "read", 0xffff_ffff) as u32;
                    face.write_mask = u(sv, "write", 0xffff_ffff) as u32;
                    d.front_face_stencil = face;
                    d.back_face_stencil = face;
                }
                Some(
                    self.device
                        .new_depth_stencil_state(&d)
                        .map_err(|e| format!("depth-stencil: {e}"))?,
                )
            }
            None => None,
        };
        if let Some(ds) = &depth_stencil {
            enc.set_depth_stencil_state(ds)
                .map_err(|e| format!("depth-stencil: {e}"))?;
        }
        let raster: Vec<u64> = dv
            .get("raster")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .collect();
        if let [cull, winding, fill] = raster[..] {
            enc.set_cull_mode(CullMode::try_from(cull as u32).map_err(|e| format!("cull: {e:?}"))?);
            enc.set_front_facing_winding(
                Winding::try_from(winding as u32).map_err(|e| format!("winding: {e:?}"))?,
            );
            enc.set_triangle_fill_mode(
                TriangleFillMode::try_from(fill as u32).map_err(|e| format!("fill: {e:?}"))?,
            )
            .map_err(|e| format!("fill: {e}"))?;
        }
        let primitive: PrimitiveType = raw(dv, "primitive", 3)?;
        match dv.get("indexType").and_then(Value::as_u64) {
            Some(k) => {
                let indices = self
                    .device
                    .new_buffer(64, ResourceOptions::shared())
                    .map_err(|e| format!("index buffer: {e}"))?;
                enc.draw_indexed_primitives(
                    primitive,
                    3,
                    IndexType::try_from(k as u32).map_err(|e| format!("index type: {e:?}"))?,
                    &indices,
                    0,
                    1,
                    0,
                    0,
                )
                .map_err(|e| format!("draw: {e}"))?;
                keep.0.push(indices);
            }
            None => enc
                .draw_primitives(primitive, 0, 3, 1, 0)
                .map_err(|e| format!("draw: {e}"))?,
        }
        enc.end_encoding().map_err(|e| format!("end: {e}"))?;
        cb.commit().map_err(|e| format!("commit: {e}"))?;
        cb.wait_until_completed_for(std::time::Duration::from_secs(60))
            .map_err(|_| "the command buffer did not complete".to_string())?;
        drop(keep);
        match cb.status() {
            CommandBufferStatus::Completed => Ok(()),
            other => Err(format!(
                "the command buffer ended {other:?}: {:?}",
                cb.error()
            )),
        }
    }

    /// A texture as the capture recorded it bound: over a buffer at its offset and rows where it
    /// was (a surface's or a buffer's), else its own; a view is made as a texture of its own.
    fn texture(&self, tv: &Value, keep: &mut Vec<Buffer>) -> Result<Texture, String> {
        let format: PixelFormat = raw(tv, "format", 0)?;
        let width = u(tv, "width", 1).clamp(1, 4096);
        let height = u(tv, "height", 1).clamp(1, 4096);
        let mut d = TextureDescriptor::texture_2d(format, width, height, false);
        d.texture_type = raw::<TextureType>(tv, "type", 2)?;
        d.depth = u(tv, "depth", 1).max(1);
        d.mipmap_level_count = u(tv, "levels", 1).max(1);
        d.array_length = u(tv, "array", 1).max(1);
        d.sample_count = u(tv, "samples", 1).max(1);
        d.usage = raw::<TextureUsage>(tv, "usage", 1)?;
        d.resource_options =
            ResourceOptions::from_raw(u(tv, "options", 0)).map_err(|e| e.to_string())?;
        let over = tv.get("over").and_then(Value::as_str);
        let row = u(tv, "row", 0);
        if over.is_some() && row > 0 {
            let offset = u(tv, "offset", 0);
            let length = offset + row * height + 4096;
            let buffer = self
                .device
                .new_buffer(length, d.resource_options)
                .map_err(|e| format!("buffer: {e}"))?;
            let texture = buffer
                .new_texture(&d, offset, row)
                .map_err(|e| format!("texture over buffer: {e}"))?;
            keep.push(buffer);
            return Ok(texture);
        }
        self.device.new_texture(&d).map_err(|e| e.to_string())
    }

    fn sampler(&self, sv: &Value) -> Result<SamplerState, String> {
        let mut d = SamplerDescriptor::default();
        d.min_filter = raw::<SamplerMinMagFilter>(sv, "min", 0)?;
        d.mag_filter = raw::<SamplerMinMagFilter>(sv, "mag", 0)?;
        d.mip_filter = raw::<SamplerMipFilter>(sv, "mip", 0)?;
        d.s_address_mode = raw::<SamplerAddressMode>(sv, "s", 0)?;
        d.t_address_mode = raw::<SamplerAddressMode>(sv, "t", 0)?;
        d.r_address_mode = raw::<SamplerAddressMode>(sv, "r", 0)?;
        d.border_color = raw::<SamplerBorderColor>(sv, "border", 0)?;
        d.normalized_coordinates = b(sv, "normalized", true);
        d.lod_min_clamp = sv.get("lodMin").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        d.lod_max_clamp = sv
            .get("lodMax")
            .and_then(Value::as_f64)
            .unwrap_or(f64::from(f32::MAX)) as f32;
        d.lod_average = b(sv, "lodAverage", false);
        d.max_anisotropy = u(sv, "maxAniso", 1).max(1) as u32;
        d.compare_function = raw::<CompareFunction>(sv, "compare", 0)?;
        self.device.new_sampler_state(&d).map_err(|e| e.to_string())
    }

    /// Waits until the runtime has said nothing for 300 ms (at most 10 s): a draw whose own
    /// pipeline is not made yet runs on its state's predicted one while its own is made on a pool
    /// thread, which reports through the creation log when it is done.
    fn settle(&self) {
        let count = || self.creation_log.lock().map(|l| l.len()).unwrap_or(0);
        let began = Instant::now();
        let (mut seen, mut quiet) = (count(), Instant::now());
        while began.elapsed() < std::time::Duration::from_secs(10)
            && quiet.elapsed() < std::time::Duration::from_millis(300)
        {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let now = count();
            if now != seen {
                (seen, quiet) = (now, Instant::now());
            }
        }
    }

    /// Keeps a creation's log with the runtime's creation-log lines after the driver's output
    /// (`--keep-logs`), or removes it.
    fn finish_log(&self, path: &Path) {
        if !self.keep_logs {
            let _ = std::fs::remove_file(path);
            return;
        }
        if let (Ok(mut f), Ok(lines)) = (
            std::fs::OpenOptions::new().append(true).open(path),
            self.creation_log.lock(),
        ) {
            for line in lines.iter() {
                let _ = writeln!(f, "metal2vulkan: {line}");
            }
        }
    }

    /// Makes function `id` once, timing its specialization.
    fn function(&mut self, id: &str, functions: &[(String, Value)]) -> Result<(), String> {
        if self.functions.contains_key(id) {
            return Ok(());
        }
        let fv = functions
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, v)| v)
            .ok_or_else(|| format!("no function {id}"))?;
        let name = s(fv, "name")?;
        let library = match fv.get("library").and_then(Value::as_str) {
            Some(lid) => self.libraries.iter().find(|(l, _, _)| l == lid),
            None => self
                .libraries
                .iter()
                .find(|(_, _, names)| names.iter().any(|n| n == name)),
        }
        .map(|(_, l, _)| l)
        .ok_or_else(|| format!("function {id} ({name}): in no library"))?;
        let mut constants = FunctionConstantValues::defaults();
        for c in fv
            .get("constants")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let ty: DataType = raw(c, "type", 0)?;
            let value = FunctionConstantValue::new(ty, hex(s(c, "hex")?)?);
            constants = match c.get("name").and_then(Value::as_str) {
                Some(n) => constants.with_name(n, value),
                None => constants.with_index(
                    u32::try_from(u(c, "index", 0)).map_err(|e| e.to_string())?,
                    value,
                ),
            };
        }
        let (wall0, cpu0) = (Instant::now(), thread_cpu());
        let function = match constants.is_empty() {
            true => library.new_function_with_name(name),
            false => library.new_function_with_name_constant_values(name, &constants),
        }
        .map_err(|error| format!("function {id} ({name}): {error}"))?;
        let (wall, cpu) = (wall0.elapsed(), thread_cpu() - cpu0);
        self.function_us.insert(
            id.to_string(),
            (wall.as_micros(), cpu.as_micros(), name.to_string()),
        );
        self.functions.insert(id.to_string(), function);
        Ok(())
    }

    fn stage(&self, id: &str) -> Result<StageSource, String> {
        let f = self
            .functions
            .get(id)
            .ok_or_else(|| format!("no function {id}"))?;
        Ok(StageSource::Function(f.function_descriptor()))
    }

    fn render(&self, pv: &Value) -> Result<RenderPipelineState, String> {
        let mut d = RenderPipelineDescriptor::new(self.stage(s(pv, "vertex")?)?);
        d.fragment = match pv.get("fragment").and_then(Value::as_str) {
            Some(fid) => Some(self.stage(fid)?),
            None => None,
        };
        for (k, c) in pv
            .get("colors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if c.is_null() || k >= 8 {
                continue;
            }
            let mut color = ColorAttachmentDescriptor::new(raw::<PixelFormat>(c, "format", 0)?);
            color.blending_enabled = b(c, "blending", false);
            color.source_rgb_blend_factor = raw::<BlendFactor>(c, "srcRGB", 1)?;
            color.destination_rgb_blend_factor = raw::<BlendFactor>(c, "dstRGB", 0)?;
            color.rgb_blend_operation = raw::<BlendOperation>(c, "rgbOp", 0)?;
            color.source_alpha_blend_factor = raw::<BlendFactor>(c, "srcA", 1)?;
            color.destination_alpha_blend_factor = raw::<BlendFactor>(c, "dstA", 0)?;
            color.alpha_blend_operation = raw::<BlendOperation>(c, "alphaOp", 0)?;
            color.write_mask = raw::<ColorWriteMask>(c, "mask", 15)?;
            d.color_attachments[k] = Some(color);
        }
        let format = |key: &str| -> Result<Option<PixelFormat>, String> {
            match u(pv, key, 0) {
                0 => Ok(None),
                _ => raw::<PixelFormat>(pv, key, 0).map(Some),
            }
        };
        d.depth_format = format("depth")?;
        d.stencil_format = format("stencil")?;
        d.raster_sample_count = u(pv, "samples", 1).max(1) as u32;
        d.alpha_to_coverage = b(pv, "alphaToCoverage", false);
        d.rasterization_enabled = b(pv, "rasterization", true);
        d.input_primitive_topology = raw::<PrimitiveTopologyClass>(pv, "topology", 0)?;
        if let Some(vd) = pv.get("vertexDescriptor").filter(|v| v.is_object()) {
            let mut inputs = VertexDescriptor::new();
            for a in vd
                .get("attributes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                inputs.attributes.insert(
                    u(a, "index", 0) as u32,
                    VertexAttributeDescriptor::new(
                        raw::<VertexFormat>(a, "format", 0)?,
                        u(a, "offset", 0) as u32,
                        u(a, "buffer", 0) as u32,
                    ),
                );
            }
            for l in vd
                .get("layouts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let mut layout = VertexBufferLayoutDescriptor::new(Some(u(l, "stride", 0) as u32));
                layout.step_function = raw::<VertexStepFunction>(l, "step", 1)?;
                layout.step_rate = u(l, "rate", 1) as u32;
                inputs.layouts.insert(u(l, "index", 0) as u32, layout);
            }
            d.vertex_descriptor = Some(inputs);
        }
        self.device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
            .map_err(|e| e.to_string())
    }

    fn compute(&self, pv: &Value) -> Result<(), String> {
        let fid = s(pv, "function")?;
        let function = self
            .functions
            .get(fid)
            .ok_or_else(|| format!("no function {fid}"))?;
        self.device
            .new_compute_pipeline_state_with_function(function, &PipelineOptions::default())
            .map(drop)
            .map_err(|e| e.to_string())
    }
}
