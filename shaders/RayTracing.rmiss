#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#include "lib/RayTracingCommons.glsl"
#include "lib/UniformBufferObject.glsl"

layout(binding = UNIFORM_BIND) readonly uniform UniformBufferObjectStruct { UniformBufferObject Camera; };
layout(binding = DLIGHT_BIND) readonly buffer Lights { Light[] lights; };

layout(binding = SKYBOX_BIND) uniform samplerCube skybox;

layout(location = 0) rayPayloadInEXT RayPayload ray;

void main()
{
	const uint num_lights = lights.length();
	vec3 light_acc = vec3(0.);
	vec3 ray_direction = normalize(gl_WorldRayDirectionEXT.xyz);
	
	if (ray.t != 0) {
		for (int i = 0; i < num_lights; i++) {
			Light li = lights[i];
			float cos = dot(normalize(li.transform.xyz), ray_direction);
			if (cos < 0.) {
				light_acc -= cos * li.color.xyz * li.intensity;
			}
		}
	}

	if (Camera.HasSky) {
		// Sky color
		const vec3 skyColor = texture(skybox, ray_direction).rgb;
		light_acc += skyColor + light_acc;
	}

	ray.hitValue = vec3(0.);
	ray.needScatter = false;
	ray.emittance = light_acc;
	ray.t = -1.;
}
