// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::fractal_compute_pipeline::FractalComputePipeline;
use crate::place_over_frame::RenderPassPlaceOverFrame;
use cgmath::Vector2;
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Queue;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::sync::GpuFuture;
use vulkano_util::renderer::{DeviceImageView, VulkanoWindowRenderer};
use vulkano_util::window::WindowDescriptor;
use winit::window::Fullscreen;
use winit::{
    dpi::PhysicalPosition,
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
};

const SENSATIVITY: f32 = 0.002;
const SPEED: f32 = 1.;

/// App for exploring Julia and Mandelbrot fractals
pub struct FractalApp {
    /// Pipeline that computes Mandelbrot & Julia fractals and writes them to an image
    fractal_pipeline: FractalComputePipeline,
    /// Our render pipeline (pass)
    pub place_over_frame: RenderPassPlaceOverFrame,
    time: Instant,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    /// Input state to handle mouse positions, continuous movement etc.
    input_state: InputState,
    orientation: quaternion::Quaternion<f32>,
    shader_data: cs::ty::Data,
}

impl FractalApp {
    pub fn new(gfx_queue: Arc<Queue>, image_format: vulkano::format::Format) -> FractalApp {
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

        FractalApp {
            fractal_pipeline: FractalComputePipeline::new(
                gfx_queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
            ),
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue,
                &memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                image_format,
            ),
            time: Instant::now(),
            frame_count: 0.0,
            avg_fps: 0.0,
            dt_sum: 0.0,
            input_state: InputState::new(),
            orientation: quaternion::id::<f32>(),
            shader_data: cs::ty::Data {
                u_view: [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., -6., 1.],
                ],
                t: 0.,
                dt: 0.,
                a: 0,
                b: 0,
                c: 0,
                d: 0,
                e: 0,
                f: 0,
                samples: 1,
                depth: 3,
                max_dist: 1e+6,
                min_dist: 1e-4,
                cameraFovAngle: (PI * 2. / 3.) as f32,
                paniniDistance: 1.,
                lensFocusDistance: 4.,
                circleOfConfusionRadius: 0.,
                exposure: 1.,
                ambience: 0.,
                scatter_t: 100.,
                scatter_bias: 1.,
                light_pos: [2., 6.89, 1.],
                light_color: [0xff as f32 / 255., 0xff as f32 / 255., 0xff as f32 / 255.],
                sphere_center: [-1., 0., 0.],
                plane_center: [0., -1., 0.],
                cylinder_center: [1., 0., 4.],
                _dummy0: [0; 4],
                _dummy1: [0; 4],
                _dummy2: [0; 4],
                _dummy3: [0; 4],
            },
        }
    }

    pub fn print_guide(&self) {
        println!(
            "\
Usage:
    WASD: Pan view
    Scroll: Zoom in/out
    Space: Toggle between Mandelbrot and Julia
    Enter: Randomize color palette
    Equals/Minus: Increase/Decrease max iterations
    F: Toggle full-screen
    Right mouse: Stop movement in Julia (mouse position determines c)
    Esc: Quit\
        "
        );
    }

    /// Run our compute pipeline and return a future of when the compute is finished
    pub fn compute(&mut self, image_target: DeviceImageView) -> Box<dyn GpuFuture> {
        self.fractal_pipeline
            .compute(image_target, self.shader_data)
    }

    /// Should the app quit? (on esc)
    pub fn is_running(&self) -> bool {
        !self.input_state.should_quit
    }

    /// Return average fps
    pub fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Delta time in milliseconds
    pub fn dt(&self) -> f32 {
        self.shader_data.dt * 1000.0
    }

    /// Update times and dt at the end of each frame
    pub fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.shader_data.dt = self.time.elapsed().as_secs_f32();
        self.shader_data.t += self.shader_data.dt;
        self.dt_sum += self.shader_data.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();
    }

    /// Updates app state based on input state
    pub fn update_state_after_inputs(&mut self, renderer: &mut VulkanoWindowRenderer) {
        // Zoom in or out
        self.shader_data.u_view[3][3] *= 1.05_f64.powf(self.input_state.scroll_delta as f64) as f32;
        let right = quaternion::rotate_vector(self.orientation, [1., 0., 0.]);
        let mv_up = Vector3::new(0., 1., 0.);
        let mv_right = Vector3::new(right[0], 0., right[2]);
        let mv_front = {
            let front = quaternion::rotate_vector(self.orientation, [0., 0., 1.]);
            Vector3::new(front[0], 0., front[2])
        };

        // Move speed scaled by zoom level
        let move_speed = SPEED * self.shader_data.dt;
        let mut delta = self.input_state.controls_move;
        delta = Matrix3::from_cols(mv_front, mv_right, mv_up) * delta;
        delta *= move_speed;
        self.shader_data.u_view[3] =
            vecmath::vec4_add(self.shader_data.u_view[3], [delta.x, delta.y, delta.z, 0.]);
        if self.input_state.mouse_delta.x != 0. && self.input_state.mouse_delta.y != 0. {
            let d = self.input_state.mouse_delta;

            let q_x = quaternion::axis_angle::<f32>(
                [0., 1., 0.],
                d.x as f32 / self.shader_data.u_view[3][3] * SENSATIVITY,
            );
            let q_y = quaternion::axis_angle::<f32>(
                right,
                d.y as f32 / self.shader_data.u_view[3][3] * SENSATIVITY,
            );
            let q_z = quaternion::rotation_from_to(right, [mv_right.x, mv_right.y, mv_right.z]);
            self.orientation = quaternion::mul(q_x, self.orientation);
            self.orientation = quaternion::mul(q_y, self.orientation);
            self.orientation = quaternion::mul(q_z, self.orientation);

            let rot = quat_to_mat3(self.orientation);

            (0..3).for_each(|i| {
                (0..3).for_each(|j| {
                    self.shader_data.u_view[i][j] = rot[i][j];
                })
            });
        }
        // Toggle full-screen
        if self.input_state.toggle_full_screen {
            renderer.toggle_full_screen();
        }
        self.input_state.mouse_delta = Vector2::new(0.0, 0.0);
    }

    /// Update input state
    pub fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.input_state.handle_input(window_size, event);
    }

    /// reset input state at the end of frame
    pub fn reset_input_state(&mut self) {
        self.input_state.reset()
    }
}

fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}

/// Just a very simple input state (mappings).
/// Winit only has Pressed and Released events, thus continuous movement needs toggles.
/// Panning is one of those where continuous movement feels better.
struct InputState {
    pub window_size: [f32; 2],
    pub toggle_full_screen: bool,
    pub should_quit: bool,
    pub scroll_delta: f32,
    pub mouse_pos: Vector2<f32>,
    pub mouse_delta: Vector2<f32>,
    pub controls_move: Vector3<f32>,
}

impl InputState {
    fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            toggle_full_screen: false,
            should_quit: false,
            scroll_delta: 0.0,
            mouse_pos: Vector2::new(0.0, 0.0),
            mouse_delta: Vector2::new(0.0, 0.0),
            controls_move: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    // fn normalized_mouse_pos(&self) -> Vector2<f32> {
    //     Vector2::new(
    //         (self.mouse_pos.x / self.window_size[0] as f32).clamp(0.0, 1.0),
    //         (self.mouse_pos.y / self.window_size[1] as f32).clamp(0.0, 1.0),
    //     )
    // }

    // Resets values that should be reset. All incremental mappings and toggles should be reset.
    fn reset(&mut self) {
        *self = InputState {
            scroll_delta: 0.0,
            toggle_full_screen: false,
            ..*self
        }
    }

    fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.window_size = window_size;
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::KeyboardInput { input, .. } => self.on_keyboard_event(input),
                WindowEvent::MouseInput { state, button, .. } => {
                    self.on_mouse_click_event(*state, *button)
                }
                WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
                WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
                _ => {}
            }
        }
    }

    /// Match keyboard event to our defined inputs
    fn on_keyboard_event(&mut self, input: &KeyboardInput) {
        if let Some(key_code) = input.virtual_keycode {
            match key_code {
                VirtualKeyCode::Escape => self.should_quit = state_is_pressed(input.state),
                VirtualKeyCode::W => {
                    if state_is_pressed(input.state) {
                        self.controls_move.x = 1_f32.min(self.controls_move.x + 1.)
                    } else {
                        self.controls_move.x = 0_f32.min(self.controls_move.x)
                    }
                }
                VirtualKeyCode::A => {
                    if state_is_pressed(input.state) {
                        self.controls_move.y = -1_f32.max(self.controls_move.y - 1.)
                    } else {
                        self.controls_move.y = 0_f32.max(self.controls_move.y)
                    }
                }
                VirtualKeyCode::S => {
                    if state_is_pressed(input.state) {
                        self.controls_move.x = -1_f32.max(self.controls_move.x - 1.)
                    } else {
                        self.controls_move.x = 0_f32.max(self.controls_move.x)
                    }
                }
                VirtualKeyCode::D => {
                    if state_is_pressed(input.state) {
                        self.controls_move.y = 1_f32.min(self.controls_move.y + 1.)
                    } else {
                        self.controls_move.y = 0_f32.min(self.controls_move.y)
                    }
                }
                VirtualKeyCode::Space => {
                    if state_is_pressed(input.state) {
                        self.controls_move.z = 1_f32.min(self.controls_move.z + 1.)
                    } else {
                        self.controls_move.z = 0_f32.min(self.controls_move.z)
                    }
                }
                VirtualKeyCode::LShift => {
                    if state_is_pressed(input.state) {
                        self.controls_move.z = -1_f32.max(self.controls_move.z - 1.)
                    } else {
                        self.controls_move.z = 0_f32.max(self.controls_move.z)
                    }
                }
                VirtualKeyCode::F => self.toggle_full_screen = state_is_pressed(input.state),
                _ => (),
            }
        }
    }

    /// Update mouse scroll delta
    fn on_mouse_wheel_event(&mut self, delta: &MouseScrollDelta) {
        let change = match delta {
            MouseScrollDelta::LineDelta(_x, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.scroll_delta += change;
    }

    /// Update mouse position
    fn on_cursor_moved_event(&mut self, pos: &PhysicalPosition<f64>) {
        self.mouse_pos = Vector2::new(pos.x as f32, pos.y as f32);
    }

    /// Update toggle julia state (if right mouse is clicked)
    fn on_mouse_click_event(&mut self, state: ElementState, mouse_btn: winit::event::MouseButton) {
        if mouse_btn == MouseButton::Right {
            self.toggle_c = state_is_pressed(state)
        }
    }
}

fn quat_to_mat3((w, r): quaternion::Quaternion<f32>) -> vecmath::Matrix3<f32> {
    let mut mat = [[0.; 3]; 3];

    let del = |i, j| (i == j) as i32 as f32;
    let eps = |i, j, k| {
        ((i as i32 - j as i32) * (j as i32 - k as i32) * (k as i32 - i as i32)) as f32 / 2.
    };

    let mut cross_mat = [[0.; 3]; 3];

    (0..3).for_each(|m| {
        (0..3).for_each(|k| {
            cross_mat[m][k] = (0..3)
                .map(|i| (0..3).map(|j| del(m, i) * eps(i, j, k) * r[j]).sum::<f32>())
                .sum::<f32>();
        })
    });

    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            mat[i][j] = del(i, j)
                - 2. * (w * cross_mat[i][j]
                    - (0..3)
                        .map(|k| cross_mat[i][k] * cross_mat[k][j])
                        .sum::<f32>());
        })
    });

    mat
}
