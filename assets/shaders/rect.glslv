#version 460

layout(location = 0) in vec2 pos;
layout(location = 0) out vec2 _pos;

void main() {
  gl_Position = vec4(pos, 0.0, 1.0);
  _pos = (pos + 1) / 2;
}