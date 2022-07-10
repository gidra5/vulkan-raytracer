#version 460
#include "./utils/const.glsl"

const uint TYPE_DIFFUSE = 0x00000001u;
const uint TYPE_REFRACTIVE = 0x00000002u;
const uint TYPE_SUBSUFRACE = 0x00000004u;

layout(location = 0) in vec3 ray_dir;

layout(location = 1) out vec4 frag_color;

layout(binding = 0) uniform samplerCube skybox;

layout(set = 0, binding = 1) uniform Data {
  mat4 u_proj;
  mat4 u_view;
  vec2 u_resolution;
  float t;
  float dt;
  uint a;
  uint b;
  uint c;
  uint d;
  uint e;
  uint f;

  uint samples;
  uint depth;

  float max_dist;
  float min_dist;

  float cameraFovAngle;
  float paniniDistance;
  float lensFocusDistance;
  float circleOfConfusionRadius;
  float exposure;
  float ambience;
  float scatter_t;
  float scatter_bias;

  vec3 light_pos;
  vec3 light_color;
  vec3 sphere_center;
  vec3 plane_center;
  vec3 cylinder_center;
};

const mat3 triangle_pts = mat3(
  vec3(0.),
  vec3(0., 1., 0.),
  vec3(1., 1., 0.)
);
const vec3 sun_color = vec3(0x92, 0x97, 0xC4) / 0xff * 0.9;
// const vec3 sun_dir = normalize(vec3(1., 1., -1.));
const vec3 sun_dir = normalize(vec3(10., 1., -1.));
  // vec3 dir = normalize(vec3(0.01, 1., -0.01));

struct Ray {
  vec3 pos; // Origin
  vec3 dir; // Direction (normalized)
  vec3 throughput;
};

struct Light {
  vec3 pos; 
  vec3 color; 
};

struct Material {
  vec3 color;
  vec3 emission;
  uint type;

  float specularChance;
  float roughness;
  vec3 specularColor;

  float refraction_chance;    // percent chance of doing a refractive transmission
  float refraction_roughness; // how rough the refractive transmissions are
  vec3  refraction_color;     // absorption for beer's law  

  // TYPE_REFLECTIVE
  float reflection_fuzz;

  // TYPE_REFRACTIVE
  float refraction_index;
};

struct Hit {
  vec3 normal;
  float dist;
  bool hit;
  Material material;
};
#include "./utils/rand.comp"
#include "./utils/raytracePrimitives.comp"
#include "./utils/samplers.comp"
#include "./utils/utils.comp"



// float raw_ds(vec3 p) {
//   float d = max_dist;

//   d = min(d, sphere(p - sphere_center, 1.));
//   d = min(d, plane(p - plane_center, -normalize(vec3(0., 1., 0.))));
//   d = min(d, cylinder(p - cylinder_center, vec3(0., 1., 0.), 1.));

//   d = min(d, triangle(p - vec3(2., 2., 2.), triangle_pts[0], triangle_pts[1], triangle_pts[2]));

//   return d;
// }

// float dist_scene(Ray ray) {
//   vec3 p = ray.pos;
//   vec3 dir = ray.dir;
//   float d = max_dist;

//   // d = min(d, sphere(p - sphere_center, dir, 1.));
//   if (can_sphere(Ray(p - sphere_center, dir), 1.))
//     d = min(d, sphere(p - sphere_center, 1.));

//   d = min(d, plane(Ray(p - plane_center, dir), vec3(0., 1., 0.)));
//   // if (can_plane(p - plane_center, dir, vec3(0., 1., 0.)))
//   // d = min(d, plane(p - plane_center, vec3(0., 1., 0.)));

//   if (can_cylinder(Ray(p - cylinder_center, dir), vec3(0., 1., 0.), 1.))
//     d = min(d, cylinder(p - cylinder_center, vec3(0., 1., 0.), 1.));

//   // if (can_triangle(dir, p - vec3(2., 0., 2.), triangle_pts[0], triangle_pts[1], triangle_pts[2]))
//     // d = min(d, triangle(p - vec3(2., 0., 2.), triangle_pts[0], triangle_pts[1], triangle_pts[2]));

//   return d;
// }

// vec3 dist_scene_gradient(vec3 p) {
//   float d = raw_ds(p);
//   vec2 e = vec2(min_dist, 0);

//   return (vec3(raw_ds(p + e.xyy), raw_ds(p + e.yxy), raw_ds(p + e.yyx)) - d) / min_dist;
// }

// vec3 raymarch(Ray ray, out bool hit) {
//   vec3 origin = ray.pos;
//   vec3 dir = ray.dir;
//   vec3 p = origin;
//   hit = false;

//   float d = 0.;
//   for (int steps = 0; steps < max_steps && d < max_dist && !hit; ++steps) {
//     float ds = dist_scene(Ray(p, dir));
//     // float ds = raw_ds(p);

//     d += ds;
//     p += dir * ds;

//     hit = ds < min_dist;
//   }

//   return p;
// }

void resetMaterial(out Material m) {
  m.roughness = 1.;
  m.color = vec3(1.);
  m.emission = vec3(0.);
  m.specularChance = 0.0;
  m.specularColor = vec3(0.);
  m.refraction_index = 1.0;
  m.refraction_chance = 0.;
  m.refraction_color = vec3(0.0);
  m.refraction_roughness = 0.0;
}

Hit scene(in Ray ray) {
  Hit hitObj;
  hitObj.dist = max_dist;
  float d2;
  resetMaterial(hitObj.material);

  // d2 = iTorus(ray.pos - plane_center - vec3(-1., 5., -1.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec2(1., 0.5));
  // hitObj.hit = hitObj.hit || d2 < hitObj.dist;
  // if (d2 < hitObj.dist) {
  //   hitObj.material.color = vec3(1., 1., 1.);
  //   hitObj.material.type = TYPE_SUBSUFRACE;
  // }
  // hitObj.dist = min(hitObj.dist, d2);

  d2 = iPlane(ray.pos - plane_center, ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(0., 1., 0.), 0.);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.type = TYPE_DIFFUSE;
  }

  d2 = iSphere(ray.pos - sphere_center, ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 1.);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(0., 1., 0.);
    hitObj.material.specularColor = hitObj.material.color;
    hitObj.material.type = TYPE_DIFFUSE;
    hitObj.material.roughness = 0.05;
    hitObj.material.specularChance = 1.;
  }

  d2 = iSphere(ray.pos - sphere_center - vec3(0., 0., -2.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 1.);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(0., 0.5, 0.5);
    hitObj.material.type = TYPE_DIFFUSE;
  }

  d2 = iSphere(ray.pos - sphere_center - vec3(1., 2., 0.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 0.5);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.type = TYPE_REFRACTIVE;
    hitObj.material.refraction_index = 1.5;
  }

  for (int i = 0; i < 10; ++i) {
    d2 = iSphere(ray.pos - sphere_center - vec3(1.5 + i, -.5, 1.5), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 0.5);
    if (d2 < hitObj.dist) {
      hitObj.hit = true;
      hitObj.dist = d2;
      resetMaterial(hitObj.material);

      // hitObj.material.color = vec3(0.9f, 0.25f, 0.25f);
      hitObj.material.color = vec3(1.);
      hitObj.material.specularChance = 0.02f;
      hitObj.material.roughness = 0.;
      hitObj.material.specularColor = vec3(0.8f);
      hitObj.material.refraction_chance = 1.0f;
      hitObj.material.refraction_roughness = 0.0f;

      hitObj.material.type = TYPE_REFRACTIVE;
      hitObj.material.refraction_index = 1. + log(1 + exp(2. * (i - 3))) * 0.5;
    }

    d2 = iSphere(ray.pos - sphere_center - vec3(1.5 + i, -.5, 2.5), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 0.5);
    if (d2 < hitObj.dist) {
      hitObj.hit = true;
      hitObj.dist = d2;
      resetMaterial(hitObj.material);

      // hitObj.material.color = vec3(0.9f, 0.25f, 0.25f);
      hitObj.material.color = vec3(1.);
      hitObj.material.specularChance = 0.02f;
      hitObj.material.roughness = 0.;
      hitObj.material.specularColor = vec3(1.0f, 1.0f, 1.0f) * 0.8f;
      hitObj.material.refraction_chance = 1.0f;
      hitObj.material.refraction_roughness = 0.0f; 

      hitObj.material.type = TYPE_REFRACTIVE;
      hitObj.material.refraction_index = exp(-i * 0.6) / (1 + exp(-i * 0.6));
    }

    d2 = iSphere(ray.pos - sphere_center - vec3(1.5 + i, -.5, 3.5), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 0.5);
    if (d2 < hitObj.dist) {
      hitObj.hit = true;
      hitObj.dist = d2;
      resetMaterial(hitObj.material);

      // hitObj.material.color = vec3(0.9f, 0.25f, 0.25f);
      hitObj.material.color = vec3(1.);
      hitObj.material.specularChance = 0.02f;
      hitObj.material.roughness = 0.;
      hitObj.material.specularColor = vec3(1.0f, 1.0f, 1.0f) * 0.8f;
      hitObj.material.refraction_chance = 1.0f;
      hitObj.material.refraction_roughness = 0.0f;
      
      hitObj.material.type = TYPE_REFRACTIVE;
      hitObj.material.refraction_index = -log(1 + exp(2. * (i - 3))) * 0.5;
    }
  }

  for (int i = 0; i <= 6; ++i) {
    for (int j = 0; j <= 6; ++j) {
      d2 = iSphere(ray.pos - sphere_center - vec3(2.5 + j, -.5, -1.5 - i), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 0.5);
      if (d2 < hitObj.dist) {
        hitObj.hit = true;
        hitObj.dist = d2;
        resetMaterial(hitObj.material);

        hitObj.material.color = vec3(1., 1., 1.);
        hitObj.material.type = TYPE_DIFFUSE;
        hitObj.material.roughness = i / 6.;
        hitObj.material.specularChance = j / 6.;
        hitObj.material.specularColor = vec3(0.8, 0.2, 0.8);
      }
    }
  }

  d2 = iSphere(ray.pos - sphere_center - vec3(0., 0., 2.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, 1.);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.specularColor = hitObj.material.color;
    hitObj.material.type = TYPE_DIFFUSE;
    hitObj.material.specularChance = 1.;
  }

  d2 = iPlane(ray.pos - plane_center - vec3(-3., 0., 0.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(1., 0., 0.), 0.);
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.type = TYPE_DIFFUSE;
  }

  d2 = iBox(ray.pos - plane_center - vec3(0., 0., 5.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(20., 9., 0.1));
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 0.);
    hitObj.material.type = TYPE_DIFFUSE;
    // hitObj.material.type = TYPE_REFRACTIVE;
    // hitObj.material.refraction_index = 1.31;
  }

  d2 = iBox(ray.pos - plane_center - vec3(0., 0., -7.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(20., 9., 0.1));
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(0., 1., 1.);
    hitObj.material.type = TYPE_DIFFUSE;
    // hitObj.material.type = TYPE_REFRACTIVE;
    // hitObj.material.refraction_index = 1.31;
  }

  d2 = iBox(ray.pos - plane_center - vec3(0., 8., 0.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(20., 0.1, 20.));
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.type = TYPE_DIFFUSE;
  }

  d2 = iBox(ray.pos - light_pos, ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(1., 0.0, 1.));
  if (d2 < hitObj.dist) {
    hitObj.hit = true;
    hitObj.dist = d2;
    resetMaterial(hitObj.material);

    hitObj.material.emission = light_color;
    hitObj.material.type = TYPE_DIFFUSE;
  }

  d2 = iBox(ray.pos - plane_center - vec3(19., -13., 0.), ray.dir, vec2(min_dist, hitObj.dist), hitObj.normal, vec3(0.1, 20., 20.));
  hitObj.hit = hitObj.hit || d2 < hitObj.dist;
  if (d2 < hitObj.dist) {
    hitObj.material.color = vec3(1., 1., 1.);
    hitObj.material.type = TYPE_DIFFUSE;
  }
  hitObj.dist = min(hitObj.dist, d2);

  return hitObj;
}







float in_shadow(in Ray ray, in float mag_sq) {
  Hit hitObj = scene(ray);
  bool hit = hitObj.hit;
  float ds = hitObj.dist;

  return !hit || ds * ds >= mag_sq ? 1. : 0.;
}


Light sample_light(in vec2 _p) {
  vec2 p = 2. * sample_insquare(_p);
  return Light(light_pos + vec3(p.x, 0., p.y), 4. * light_color);
}

vec3 sun_light_col(in vec3 dir, in vec3 norm) {
  return max(dot(dir, norm), 0.) * sun_color;
}

float light_vis(in vec3 pos, in vec3 dir, in float mag_sq) {
  return in_shadow(Ray(pos, dir, vec3(1.)), mag_sq);
}

float light_vis(in vec3 pos, in vec3 dir) {
  return in_shadow(Ray(pos, dir, vec3(1.)), 0.99 / (min_dist * min_dist));
}

vec3 _light(in vec3 pos, in vec3 norm, in Light light) {
  vec3 d = light.pos - pos;
  float mag_sq = dot(d, d);
  vec3 dir = d * inversesqrt(mag_sq);
  return light_vis(pos, dir, mag_sq) * point_light_col(pos, norm, light);
}

vec3 light(in vec3 pos, in vec3 norm, in vec2 t) {
  Light light = sample_light(t);

  return _light(pos, norm, light);
}

vec3 sun(in vec3 pos, in vec3 norm) {
  return light_vis(pos, sun_dir) * sun_light_col(sun_dir, norm) + ambience * sun_color;
}








vec3 pinholeRay(in vec2 pixel) { 
  return vec3(pixel, 1/tan(cameraFovAngle / 2.f));
}

vec3 paniniRay(in vec2 pixel) {
  float halfFOV = cameraFovAngle / 2.f;
  vec2 p = vec2(sin(halfFOV), cos(halfFOV) + paniniDistance);
  float M = sqrt(dot(p, p));
  float halfPaniniFOV = atan(p.x, p.y);
  vec2 hvPan = pixel * vec2(halfPaniniFOV, halfFOV);
  float x = sin(hvPan.x) * M;
  float z = cos(hvPan.x) * M - paniniDistance;
  // float y = tan(hvPan.y) * (z + verticalCompression);
  float y = tan(hvPan.y) * (z + pow(max(0., (3. * cameraFovAngle/PI - 1.) / 8.), 0.92));

  return vec3(x, y, z);
}

void thinLensRay(in vec3 dir, in vec2 uv, out Ray ray) {
  ray.pos = vec3(uv * circleOfConfusionRadius, 0.f);
  ray.dir = normalize(dir * lensFocusDistance - ray.pos);
}





void main() {
  vec3 _frag_color = vec3(0.);
  for (int x = 0; x < int(samples); ++x) {
    vec2 subpixel = random_0t1_2(vec_to_float(gl_FragCoord.xy), x);
    vec2 uv = (2. * (gl_FragCoord.xy + subpixel) - u_resolution) / u_resolution.x;
    vec3 rayDirection = normalize(paniniRay(uv));
    // vec3 rayDirection = normalize(pinholeRay(uv));
    float uv_seed = vec_to_float(uv);
    
    Ray ray;
    thinLensRay(rayDirection, sample_incircle(random_0t1_2(uv_seed, x)), ray);

    vec4 ray_pos = u_view * vec4(ray.pos, 1.);
    ray.pos = ray_pos.xyz;
    ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
    ray.dir = (u_view * vec4(ray.dir, 0.)).xyz;

    vec3 indirect_color = vec3(0.);
    ray.throughput = vec3(1.);

    for (int i = 0; i < int(depth) + 1; ++i) {
      float seed = x + i;
      Hit hitObj = scene(ray);
      Material material = hitObj.material;
      float t = hitObj.dist;
      if (!hitObj.hit) break;
      float cos_angle = -dot(hitObj.normal, ray.dir);
      int incoming = int(sign(cos_angle));
      vec3 pos = ray.pos + t * ray.dir;
      vec3 dir;
      vec3 col;
      float scatter_t = -log(random_0t1(uv_seed, x)) * scatter_t;

      if (t < scatter_t) {
        float _r1 = random_0t1(uv_seed, x);
        if (3. * _r1 < 1.) {
          // if ((material.type & TYPE_REFRACTIVE) != 0u) {
          //   float cos_angle = -dot(hitObj.normal, ray.dir);
          //   int incoming = int(sign(cos_angle));
          //   float index = incoming > 0. ? 1. / material.refraction_index : material.refraction_index;
          //   vec3 normal = incoming * hitObj.normal;
          //   dir = refract(ray.dir, normal, index);

          //   if (dir == vec3(0.) || (abs(material.refraction_index) > 1. && abs(index) < 1 && _reflectance(cos_angle, -dot(dir, normal), index) > random_0t1(uv_seed, x)))
          //     dir = reflect(ray.dir, normal);

          //   pos = pos + min_dist * dir;
          // } 
          // else 
          // if ((material.type & TYPE_SUBSUFRACE) != 0u) {
          //   vec3 d = sample_sphere(random_0t1_2(uv_seed, x));
          //   float cos_angle = -dot(hitObj.normal, ray.dir);
          //   if (cos_angle > 0.) {
          //     float scatter_distance = pow(10., int(a) - 1);
          //     float scatter_t = -log(random_0t1(uv_seed, x)) * scatter_distance;
          //     if (scatter_t < t && i < int(depth)) {
          //       t = scatter_t;
          //       dir = d;
          //       pos = ray.pos + t * ray.dir;
          //       ray.throughput *= exp(-t / scatter_distance);
          //     } else {
          //       dir = sign(dot(hitObj.normal, d)) * d;
          //     }
          //   } else {
          //     dir = -sign(dot(hitObj.normal, d)) * d;
          //   }
          //   pos = pos + min_dist * dir;
          // } else {
            // vec3 diffuse_out = normalize(sample_sphere(random_0t1_2(uv_seed, x)) + hitObj.normal);
            // vec3 diffuse_in = normalize(sample_sphere(random_0t1_2(uv_seed, x)) + hitObj.normal);

            // float cos_angle = -dot(hitObj.normal, ray.dir);
            // float index = cos_angle > 0. ? 1. / material.refraction_index : material.refraction_index;
            // vec3 normal = sign(cos_angle) * hitObj.normal;
            // vec3 _refracted = refract(ray.dir, normal, index);
            // if (_refracted == vec3(0.) || (abs(material.refraction_index) > 1. && abs(index) < 1 && _reflectance(cos_angle, -dot(_refracted, normal), index) > random_0t1(uv_seed, x)))
            //   _refracted = reflect(ray.dir, normal);
            // vec3 refracted = normalize(mix(_refracted, sign(cos_angle) * diffuse_in, material.refraction_roughness));

            // float chance = 1.;
            // float specular_chance = material.specularChance;
            // float refraction_chance = material.refraction_chance;
            // if (specular_chance > 0.) {
            //   specular_chance = mix(specular_chance, 1., _reflectance(cos_angle, -dot(_refracted, sign(cos_angle) * hitObj.normal), index));

            //   float chanceMultiplier = (1.0f - specular_chance) / (1.0f - material.specularChance);
            //   refraction_chance *= chanceMultiplier;
            //   // diffuseChance *= chanceMultiplier;
            // }

            // col += material.emission;

            // float _r2 = random_0t1(uv_seed, x);
            // bool is_specular = _r2 < specular_chance;
            // bool is_refraction = !is_specular && _r2 < specular_chance + refraction_chance;
            // ray.throughput *= mix(material.color, material.specularColor, float(is_specular)); 
            // chance = is_specular ? specular_chance : is_refraction ? refraction_chance : 1. - (specular_chance + refraction_chance);

            // vec3 reflected = reflect(ray.dir, hitObj.normal);
            // vec3 specular = normalize(mix(reflected, diffuse_out, material.roughness));
            // dir = mix(diffuse_out, specular, float(is_specular));
            // // dir = mix(dir, refracted, float(is_refraction));
            // ray.throughput /= max(chance, min_dist);

            // if (is_refraction) pos = pos - min_dist * hitObj.normal;
            // if (is_specular) pos = pos + min_dist * hitObj.normal;
          // }

          

            // float index = incoming > 0. ? 1. / material.refraction_index : material.refraction_index;
            // vec3 normal = incoming * hitObj.normal;

            // vec3 refracted = refract(ray.dir, normal, index);
            // vec3 reflected = reflect(ray.dir, normal);
            vec3 diffuse = normalize(sample_sphere(random_0t1_2(uv_seed, x)) + hitObj.normal);

            // bool is_reflected = _reflectance(cos_angle, -dot(refracted, normal), index) > random_0t1(uv_seed, x);

            // dir = mix(
            //   mix(refracted, -diffuse, float(material.refraction_roughness > random_0t1(uv_seed, x))), 
            //   mix(reflected,  diffuse, float(material.roughness > random_0t1(uv_seed, x))), 
            //   float(is_reflected)
            // );
            // dir = mix(reflected,  diffuse, float(material.roughness > random_0t1(uv_seed, x)));
            dir = diffuse;
            ray.throughput *= material.color;
            // ray.throughput *= mix(material.color, material.specularColor, float(is_specular));
            // ray.throughput *= mix(material.color, material.specularColor, float(is_reflected));
            col += material.emission;
        } else 
        if (3. * _r1 < 2.) 
        {
          ray.throughput *= material.color; 
          col += light(pos, hitObj.normal, random_0t1_2(uv_seed, x));
        } 
        else 
        {
          ray.throughput *= material.color; 
          col += sun(pos, hitObj.normal); 
        }
      } else {
        float _r1 = random_0t1(uv_seed, x);
        pos = ray.pos + scatter_t * ray.dir;
        // atmospheric scatter
        // if (incoming > 0.) {
          if (3. * _r1 < 1.) {
            dir = normalize(sample_sphere(random_0t1_2(uv_seed, x)) + ray.dir * scatter_bias);
          } else 
          if (3. * _r1 < 2.) 
          {
            Light light = sample_light(random_0t1_2(uv_seed, x));
            vec3 d = light.pos - pos;
            float mag_sq = dot(d, d);
            col += light_vis(pos, normalize(d), mag_sq) * light.color;
          } 
          else 
          {
            col += light_vis(pos, sun_dir) * sun_color; 
          }
        // subsurface scatter
        // } else {
        //   ray.throughput *= material.color;
        //   dir = sample_sphere(random_0t1_2(uv_seed, x));
        //   pos = ray.pos + _r2 * ray.dir;
        // }
      }
      indirect_color += col * ray.throughput;

      ray = Ray(pos, dir, ray.throughput);

      // Russian Roulette
      if (all(greaterThan(vec3(random_0t1(uv_seed, x)), ray.throughput)))
        break;
  
      // Add the energy we 'lose' by randomly terminating paths
      ray.throughput /= max(ray.throughput.r, max(ray.throughput.g, ray.throughput.b));
    }

    _frag_color += indirect_color;
  }
  _frag_color /= samples;
  _frag_color *= exposure;
  _frag_color = ACESFilm(_frag_color);
  frag_color = vec4(LinearToSRGB(_frag_color), 1.);
}