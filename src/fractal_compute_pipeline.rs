// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::ImageAccess,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};
use vulkano_util::renderer::DeviceImageView;

pub struct FractalComputePipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    uniform_buffer: Arc<CpuAccessibleBuffer<cs::ty::Data>>,
}

impl FractalComputePipeline {
    pub fn set_shader_data(&mut self, shader_data: cs::ty::Data) {
        self.uniform_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                storage_buffer: true,
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            shader_data,
        )
        .unwrap();
    }

    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        shader_data: cs::ty::Data,
    ) -> FractalComputePipeline {
        let pipeline = {
            let shader = cs::load(queue.device().clone()).unwrap();
            ComputePipeline::new(
                queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };
        let uniform_buffer = CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                storage_buffer: true,
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            shader_data,
        )
        .unwrap();
        FractalComputePipeline {
            queue,
            pipeline,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            uniform_buffer,
        }
    }
    pub fn compute(&self, image: DeviceImageView) -> Box<dyn GpuFuture> {
        // Resize image if needed
        let img_dims = image.image().dimensions().width_height();
        let pipeline_layout = self.pipeline.layout();
        let desc_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, image),
                WriteDescriptorSet::buffer(1, self.uniform_buffer.clone()),
            ],
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .dispatch([img_dims[0], img_dims[1], 1])
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
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
