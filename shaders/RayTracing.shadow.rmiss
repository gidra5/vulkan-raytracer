#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#include "lib/RayTracingCommons.glsl"



layout(location = 0) rayPayloadInEXT ShadowRay ray;

void main()
{
	ray.shadow = false;
}
