use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

fn main() -> Result<_, _> {
    let event_loop = EventLoop::new();
    // (1) Call Integration::<Arc<Mutex<Allocator>>>::new() in App::new().
    let mut app = App::new(&event_loop)?;

    event_loop?.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event,
                window_id: _,
            } => {
                // (2) Call integration.handle_event(&event).
                let _response = app.egui_integration.handle_event(&event);
                match event {
                    WindowEvent::Resized(_) => {
                        app.recreate_swapchain().unwrap();
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        // (3) Call integration.recreate_swapchain(...) in app.recreate_swapchain().
                        app.recreate_swapchain().unwrap();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            }
            Event::MainEventsCleared => app.window.request_redraw(),
            Event::RedrawRequested(_window_id) => {
                // (4) Call integration.begin_frame(), integration.end_frame(&mut window),
                // integration.context().tessellate(shapes), integration.paint(...)
                // in app.draw().
                app.draw().unwrap();
            }
            _ => (),
        }
    })
    // (5) Call integration.destroy() when drop app.
}
