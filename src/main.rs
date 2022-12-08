// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use crate::app::App;
use place_over_frame::RenderPassPlaceOverFrame;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::VulkanoWindowRenderer;
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

mod app;
mod fractal_compute_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;

/// Handle events and return `bool` if we should quit
fn handle_events(
    event_loop: &mut EventLoop<()>,
    renderer: &mut VulkanoWindowRenderer,
    app: &mut App,
) -> bool {
    let mut is_running = true;
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match &event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => is_running = false,
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    renderer.resize()
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
        // Pass event for app to handle our inputs
        app.handle_input(renderer.window_size(), &event);
    });
    is_running && app.is_running()
}

/// This is an example demonstrating an application with some more non-trivial functionality.
/// It should get you more up to speed with how you can use Vulkano.
/// It contains
/// - Compute pipeline to calculate Mandelbrot and Julia fractals writing them to an image target
/// - Graphics pipeline to draw the fractal image over a quad that covers the whole screen
/// - Renderpass rendering that image over swapchain image
/// - An organized Renderer with functionality good enough to copy to other projects
/// - Simple FractalApp to handle runtime state
/// - Simple Input system to interact with the application
fn main() {
    // Create event loop
    let mut event_loop = EventLoop::new();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    let _id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "Raytracer".to_string(),
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
        |_| {},
    );

    let render_target_id = 0;
    let primary_window_renderer = windows.get_primary_renderer_mut().unwrap();
    primary_window_renderer.add_additional_image_view(
        render_target_id,
        vulkano_util::renderer::DEFAULT_IMAGE_FORMAT,
        ImageUsage {
            sampled: true,
            storage: true,
            color_attachment: true,
            transfer_dst: true,
            ..ImageUsage::empty()
        },
    );

    let gfx_queue = context.graphics_queue();
    let mut app = App::new(gfx_queue.clone());

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
        gfx_queue.device().clone(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        gfx_queue.device().clone(),
        Default::default(),
    ));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        gfx_queue.device().clone(),
    ));

    let place_over_frame = RenderPassPlaceOverFrame::new(
        gfx_queue.clone(),
        &memory_allocator,
        command_buffer_allocator,
        descriptor_set_allocator,
        primary_window_renderer.swapchain_format(),
    );

    loop {
        if !handle_events(&mut event_loop, primary_window_renderer, &mut app) {
            break;
        }

        match primary_window_renderer.window_size() {
            [w, h] => {
                // Skip this frame when minimized
                if w == 0.0 || h == 0.0 {
                    continue;
                }
            }
        }

        app.update_state_after_inputs(primary_window_renderer);

        {
            let target = primary_window_renderer.get_additional_image_view(render_target_id);
            // Start frame
            let before_pipeline_future = match primary_window_renderer.acquire() {
                Err(e) => {
                    println!("{}", e);
                    return;
                }
                Ok(future) => future,
            };

            // Compute our fractal (writes to target image). Join future with `before_pipeline_future`.
            let after_compute = app.compute(target.clone()).join(before_pipeline_future);
            // Render image over frame. Input previous future. Draw on swapchain image
            let command_buffer = place_over_frame
                .command_buffer(target, primary_window_renderer.swapchain_image_view());

            let after_renderpass_future = after_compute
                .then_execute(gfx_queue.clone(), command_buffer)
                .unwrap()
                .boxed();

            // Finish frame (which presents the view). Input last future. Wait for the future so resources are not in use
            // when we render
            primary_window_renderer.present(after_renderpass_future, true);
        }

        app.reset_input_state();
        app.update_time();
        primary_window_renderer.window().set_title(&format!(
            "{} fps: {:.2} dt: {:.2}",
            "Raytracing",
            app.avg_fps(),
            app.dt(),
        ));
    }
}
