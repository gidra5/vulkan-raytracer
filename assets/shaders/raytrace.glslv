#version 460

layout(location = 0) in vec2 pos;
layout(location = 1) out vec3 ray_dir;

void main() {
  ray_dir = vec3(pos, 0.);
}