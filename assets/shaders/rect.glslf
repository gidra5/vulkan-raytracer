#version 460

layout(location = 0) in vec2 _pos;
layout(location = 0) out vec4 f_color;

void main() {
  f_color = vec4(_pos, 0.0, 1.0);
}