// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::renderer::InterimImageView;
use std::sync::Arc;
use vulkano::buffer::CpuBufferPool;
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Queue,
    image::ImageAccess,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

pub struct FractalComputePipeline {
    gfx_queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    uniform_buffer: CpuBufferPool<cs::ty::Data>,
}

impl FractalComputePipeline {
    pub fn new(gfx_queue: Arc<Queue>) -> FractalComputePipeline {
        let pipeline = {
            let shader = cs::load(gfx_queue.device().clone()).unwrap();
            ComputePipeline::new(
                gfx_queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };
        let uniform_buffer = CpuBufferPool::new(gfx_queue.device().clone(), BufferUsage::all());
        FractalComputePipeline {
            gfx_queue,
            pipeline,
            uniform_buffer,
        }
    }
    pub fn compute(&mut self, image: InterimImageView, data: cs::ty::Data) -> Box<dyn GpuFuture> {
        // Resize image if needed
        let img_dims = image.image().dimensions().width_height();
        let pipeline_layout = self.pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, image.clone()),
                WriteDescriptorSet::buffer(1, self.uniform_buffer.next(data).unwrap()),
            ],
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([img_dims[0], img_dims[1], 1])
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.gfx_queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./assets/shaders/raytrace/raytrace.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
