//! The encode benchmark: the CPU time the runtime spends recording a compositor-like frame, per
//! frame and per draw, on the system default device. Each frame is one command buffer of one render
//! pass into a `BGRA8Unorm` target of a phone's size, of DRAWS draws that alternate two blended
//! pipeline states, each with its own vertex bytes, a vertex buffer at a moving offset and one of
//! eight sampled textures (the shape of Core Animation's layer draws), committed and waited for.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example encode_bench   # FRAMES=300 DRAWS=60 in the environment
//! FRAMES=2000 perf record -g target/release/examples/encode_bench   # where the encode time goes
//! ```
//!
//! It prints the median and p90 encode (command buffer to `end_encoding`) and commit times. Only the
//! encode window is the runtime's per-draw cost; the GPU's own time is not measured.

mod common;

use std::time::{Duration, Instant};

use metal2vulkan::metal::{BlendFactor, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    ClearColor, Device, LoadAction, PipelineOptions, RenderPassDescriptor, ResourceOptions,
    StoreAction, TextureDescriptor,
};

fn main() -> common::Result<()> {
    let env = |name: &str, default: usize| -> common::Result<usize> {
        Ok(std::env::var(name).map_or(Ok(default), |v| v.parse())?)
    };
    let (frames, draws) = (env("FRAMES", 300)?, env("DRAWS", 60)?);
    let device = Device::system_default()?;
    println!("device: {}", device.name());
    let library = common::library(&device)?;
    let vertex = library.new_function_with_name("colored_vertex")?;
    let colored = library.new_function_with_name("colored_fragment")?;
    let sampled = library.new_function_with_name("sampled_fragment")?;

    let state = |fragment: &metal2vulkan_runtime::Function| {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    };
    let states = [state(&sampled)?, state(&colored)?];

    let mut target = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 1125, 2436, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target)?;
    let mut textures = Vec::new();
    for i in 0..8 {
        let mut t = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, 64 + i * 16, 64, false);
        t.usage = TextureUsage::SHADER_READ;
        t.resource_options = ResourceOptions::private();
        textures.push(device.new_texture(&t)?);
    }
    // Six vertices' colors per draw, in one buffer the draws walk.
    let colors: Vec<u8> = (0..draws * 6)
        .flat_map(|_| [1.0f32, 1.0, 1.0, 0.5])
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let colors = device.new_buffer_with_bytes(&colors, ResourceOptions::shared())?;

    let queue = device.new_command_queue()?;
    let (mut encode, mut commit) = (Vec::new(), Vec::new());
    for frame in 0..frames {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        let started = Instant::now();
        let command_buffer = queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(&pass)?;
        for d in 0..draws {
            let s = d % 2;
            encoder.set_render_pipeline_state(&states[s])?;
            let x = (d % 10) as f32 * 0.2 - 1.0;
            let y = (d / 10) as f32 * 0.2 - 1.0 + (frame % 7) as f32 * 0.01;
            let quad: [[f32; 2]; 6] = [
                [x, y],
                [x + 0.2, y],
                [x, y + 0.2],
                [x + 0.2, y],
                [x + 0.2, y + 0.2],
                [x, y + 0.2],
            ];
            let bytes: Vec<u8> = quad.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            encoder.set_vertex_bytes(&bytes, 0)?;
            encoder.set_vertex_buffer(Some(&colors), (d * 6 * 16) as u64, 1)?;
            if s == 0 {
                encoder.set_fragment_texture(Some(&textures[d % textures.len()]), 0)?;
            }
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)?;
        }
        encoder.end_encoding()?;
        let encoded = Instant::now();
        common::run(&command_buffer)?;
        encode.push(encoded - started);
        commit.push(encoded.elapsed());
    }
    let report = |name: &str, mut times: Vec<Duration>| {
        times.sort();
        let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
        println!(
            "{name}: median {:.3} ms, p90 {:.3} ms, per draw {:.1} us",
            at(0.5).as_secs_f64() * 1e3,
            at(0.9).as_secs_f64() * 1e3,
            at(0.5).as_secs_f64() * 1e6 / draws as f64
        );
    };
    report("encode", encode);
    report("commit+wait", commit);
    Ok(())
}
