#version 460

const float PHI     = 1.61803398874989484820459; // Golden Ratio   
const float PHI_2   = 1.32471795724474602596090; // Golden Ratio 2
const float PHI_3   = 1.22074408460575947536168; // Golden Ratio 3
const float PHI_4   = 1.16730397826141868425604; // Golden Ratio 4
const float SRT     = 1.41421356237309504880169; // Square Root of Two
const float PI      = 3.14159265358979323846264;
const float E       = 2.71828182845904523536028;
const float TWO_PI  = 6.28318530717958647692528;
const float INV_PI  = 1 / PI;
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
  uint gi_reflection_depth;

  float max_dist;
  float min_dist;

  float cameraFovAngle;
  float paniniDistance;
  float lensFocusDistance;
  float circleOfConfusionRadius;
  float exposure;
  float ambience;
  float sigma_t;
  float sigma_f;

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





const float alpha_rand1 = 1/PHI;
const vec2 alpha_rand2 = 1/vec2(PHI_2, PHI_2 * PHI_2);
const vec3 alpha_rand3 = 1/vec3(PHI_3, PHI_3 * PHI_3, PHI_3 * PHI_3 * PHI_3);
const vec4 alpha_rand4 = 1/vec4(PHI_4, PHI_4 * PHI_4, PHI_4 * PHI_4 * PHI_4, PHI_4 * PHI_4 * PHI_4 * PHI_4);

float _rand1 = 0.;
vec2 _rand2 = vec2(0.);
vec3 _rand3 = vec3(0.);
vec4 _rand4 = vec4(0.);

const int base = 1<<8;

// base rng
float vec_to_float(in vec2 x) { return dot(x, vec2(PHI, PI)); }
float vec_to_float(in vec3 x) { return dot(x, vec3(PHI, PI, SRT)); }
float vec_to_float(in vec4 x) { return dot(x, vec4(PHI, PI, SRT, E)); }

float random_0t1(in float x, in float seed) {
  _rand1 = fract(_rand1 + alpha_rand1);
  return fract(sin(x * (seed + t + base) * _rand1) * SRT * 10000.0);
}
vec2 random_0t1_2(in float x, in float seed) {
  _rand2 = fract(_rand2 + alpha_rand2);
  return fract(sin(x * (seed + t + base) * _rand2) * SRT * 10000.0);
}
vec3 random_0t1_3(in float x, in float seed) {
  _rand3 = fract(_rand3 + alpha_rand3);
  return fract(sin(x * (seed + t + base) * _rand3) * SRT * 10000.0);
}
vec4 random_0t1_4(in float x, in float seed) {
  _rand4 = fract(_rand4 + alpha_rand4);
  return fract(sin(x * (seed + t + base) * _rand4) * SRT * 10000.0);
}

#define NEWTON_ITER 2
#define HALLEY_ITER 1

float cbrt( float x )
{
	float y = sign(x) * uintBitsToFloat( floatBitsToUint( abs(x) ) / 3u + 0x2a514067u );

	for( int i = 0; i < NEWTON_ITER; ++i )
    	y = ( 2. * y + x / ( y * y ) ) * .333333333;

    for( int i = 0; i < HALLEY_ITER; ++i )
    {
    	float y3 = y * y * y;
        y *= ( y3 + 2. * x ) / ( 2. * y3 + x );
    }
    
    return y;
}






//primitives are "centered" at (0, 0, 0)
float box(vec3 p, vec3 half_sides) {
  vec3 q = abs(p) - half_sides;
  return length(max(q, 0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sphere(vec3 p, float r) {
  return length(p) - r;
}

float sphere(Ray ray, float r) {
  float t = dot(ray.dir, ray.pos);
  float r_min = r + min_dist;
  float d = t + sqrt(r_min * r_min - dot(ray.pos, ray.pos) + t * t);
  // if (d < 0.) return sphere(p, r);
  // else        return d;
  if (t < 0.) return max_dist; 
  else return d;
}

float plane(vec3 p, vec3 norm) {
  return abs(dot(p, norm)) - min_dist;
}

float plane(Ray ray, vec3 norm) {
  float d1 = -dot(ray.pos, norm);
  float d2 = d1 + min_dist;
  float d3 = dot(ray.dir, norm);
  if (d1 * d3 < 0.) return max_dist;
  else              return d2 / d3;
}

float cylinder(vec3 p, vec3 dir_c, float r) {
  return length(cross(p, dir_c)) - r;
}

float triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
  vec3 ba = normalize(b - a); vec3 pa = p - a;
  vec3 cb = normalize(c - b); vec3 pb = p - b;
  vec3 ac = normalize(a - c); vec3 pc = p - c;
  mat3 t = mat3(
    cross(pc, ac),
    cross(pb, cb),
    cross(pa, ba)
  );

  vec3 d = transpose(t) * cross(ac, ba);

  ivec3 n = ivec3(lessThan(d, vec3(0.)));
  ivec3 not_n = 1 - n;
  int f = int(dot(n, n));
  if (f == 3) return abs(dot(t[2], ac)) - min_dist; // directly above triangle

  mat3 vecs = mat3(pa, pb, pc);
  int index = int(dot(f == 2 ? not_n : n, ivec3(0, 1, 2)));
  return length(f == 2 ? t[index] : vecs[index]); 
}

//can_* show if primitive could be intersected
bool can_plane(Ray ray, vec3 norm) {
  return dot(ray.dir, norm) * dot(ray.pos, norm) < 0.;
}

bool can_cylinder(Ray ray, vec3 dir_c, float r) {
  return abs(dot(ray.dir, cross(ray.pos, dir_c))) < r;
}

bool can_sphere(Ray ray, float r) {
  return length(cross(ray.pos, ray.dir)) < r;
}

bool can_triangle(Ray ray, vec3 a, vec3 b, vec3 c) {
  mat3 A = mat3(
    b - a,
    c - b,
    a - c
  );
  vec3 d = vec3(
    dot(b, b) - dot(a, a),
    dot(c, c) - dot(b, b),
    dot(a, a) - dot(c, c)
  );
  vec3 center = 2 * inverse(transpose(A)) * d;

  // return can_sphere(dir, p - center, length(a - center)) && can_plane(dir, p, normalize(cross(A[0], A[2])));
  // return can_sphere(dir, p - center, length(a - center));
  return can_plane(ray, normalize(cross(A[0], A[2])));
  // return can_cylinder(dir, p - point_ba, , max(dot(A[0], A[2]), max(dot(A[0], A[1]), dot(A[0], A[0]))) / (2. * length(A[0])));
  // return true;
}






float dot2( in vec3 v ) { return dot(v,v); }

// Plane 
float iPlane( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
              in vec3 planeNormal, in float planeDist) {
    float a = dot(rd, planeNormal);
    float d = -(dot(ro, planeNormal)+planeDist)/a;
    if (a > 0. || d < distBound.x || d > distBound.y) {
        return max_dist;
    } else {
        normal = planeNormal;
    	return d;
    }
}

// Sphere:          https://www.shadertoy.com/view/4d2XWV
float iSphere( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
               float sphereRadius ) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - sphereRadius*sphereRadius;
    float h = b*b - c;
    if (h < 0.) {
        return max_dist;
    } else {
	    h = sqrt(h);
        float d1 = -b-h;
        float d2 = -b+h;
        if (d1 >= distBound.x && d1 <= distBound.y) {
            normal = normalize(ro + rd*d1);
            return d1;
        } else if (d2 >= distBound.x && d2 <= distBound.y) { 
            normal = normalize(ro + rd*d2);            
            return d2;
        } else {
            return max_dist;
        }
    }
}

// Box:             https://www.shadertoy.com/view/ld23DV
float iBox( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal, 
            in vec3 boxSize ) {
    vec3 m = sign(rd)/max(abs(rd), 1e-8);
    vec3 n = m*ro;
    vec3 k = abs(m)*boxSize;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

	float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if (tN > tF || tF <= 0.) {
        return max_dist;
    } else {
        if (tN >= distBound.x && tN <= distBound.y) {
        	normal = -sign(rd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);
            return tN;
        } else if (tF >= distBound.x && tF <= distBound.y) { 
        	normal = -sign(rd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);
            return tF;
        } else {
            return max_dist;
        }
    }
}

// Capped Cylinder: https://www.shadertoy.com/view/4lcSRn
float iCylinder( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                 in vec3 pa, in vec3 pb, float ra ) {
    vec3 ca = pb-pa;
    vec3 oc = ro-pa;

    float caca = dot(ca,ca);
    float card = dot(ca,rd);
    float caoc = dot(ca,oc);
    
    float a = caca - card*card;
    float b = caca*dot( oc, rd) - caoc*card;
    float c = caca*dot( oc, oc) - caoc*caoc - ra*ra*caca;
    float h = b*b - a*c;
    
    if (h < 0.) return max_dist;
    
    h = sqrt(h);
    float d = (-b-h)/a;

    float y = caoc + d*card;
    if (y > 0. && y < caca && d >= distBound.x && d <= distBound.y) {
        normal = (oc+d*rd-ca*y/caca)/ra;
        return d;
    }

    d = ((y < 0. ? 0. : caca) - caoc)/card;
    
    if( abs(b+a*d) < h && d >= distBound.x && d <= distBound.y) {
        normal = normalize(ca*sign(y)/caca);
        return d;
    } else {
        return max_dist;
    }
}

// Torus:           https://www.shadertoy.com/view/4sBGDy
float iTorus( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
              in vec2 torus ) {
    // bounding sphere
    vec3 tmpnormal;
    if (iSphere(ro, rd, distBound, tmpnormal, torus.y+torus.x) > distBound.y) {
        return max_dist;
    }
    
    float po = 1.0;
    
	float Ra2 = torus.x*torus.x;
	float ra2 = torus.y*torus.y;
	
	float m = dot(ro,ro);
	float n = dot(ro,rd);

#if 1
	float k = (m + Ra2 - ra2)/2.0;
  float k3 = n;
	float k2 = n*n - Ra2*dot(rd.xy,rd.xy) + k;
  float k1 = n*k - Ra2*dot(rd.xy,ro.xy);
  float k0 = k*k - Ra2*dot(ro.xy,ro.xy);
#else
	float k = (m - Ra2 - ra2)/2.0;
	float k3 = n;
	float k2 = n*n + Ra2*rd.z*rd.z + k;
	float k1 = k*n + Ra2*ro.z*rd.z;
	float k0 = k*k + Ra2*ro.z*ro.z - Ra2*ra2;
#endif
    
#if 1
  // prevent |c1| from being too close to zero
  if (abs(k3*(k3*k3-k2)+k1) < 0.01) {
      po = -1.0;
      float tmp=k1; k1=k3; k3=tmp;
      k0 = 1.0/k0;
      k1 = k1*k0;
      k2 = k2*k0;
      k3 = k3*k0;
  }
#endif
    
    // reduced cubic
    float c2 = k2*2.0 - 3.0*k3*k3;
    float c1 = k3*(k3*k3-k2)+k1;
    float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
    
    c2 /= 3.0;
    c1 *= 2.0;
    c0 /= 3.0;

    float Q = c2*c2 + c0;
    float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
    
    float h = R*R - Q*Q*Q;
    float t = max_dist;
    
    if (h>=0.0) {
        // 2 intersections
        h = sqrt(h);
        
        float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
        float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root

        vec2 s = vec2( (v+u)+4.0*c2, (v-u)*sqrt(3.0));
    
        float y = sqrt(0.5*(length(s)+s.x));
        float x = 0.5*s.y/y;
        float r = 2.0*c1/(x*x+y*y);

        float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
        float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;

        if (t1 >= distBound.x) t=t1;
        if (t2 >= distBound.x) t=min(t,t2);
	} else {
        // 4 intersections
        float sQ = sqrt(Q);
        float w = sQ*cos( acos(-R/(sQ*Q)) / 3.0 );

        float d2 = -(w+c2); if( d2<0.0 ) return max_dist;
        float d1 = sqrt(d2);

        float h1 = sqrt(w - 2.0*c2 + c1/d1);
        float h2 = sqrt(w - 2.0*c2 - c1/d1);
        float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
        float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
        float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
        float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;

        if (t1 >= distBound.x) t=t1;
        if (t2 >= distBound.x) t=min(t,t2);
        if (t3 >= distBound.x) t=min(t,t3);
        if (t4 >= distBound.x) t=min(t,t4);
    }
    
	if (t >= distBound.x && t <= distBound.y) {
        vec3 pos = ro + rd*t;
        normal = normalize( pos*(dot(pos,pos) - torus.y*torus.y - torus.x*torus.x*vec3(1,1,-1)));
        return t;
    } else {
        return max_dist;
    }
}

// Capsule:         https://www.shadertoy.com/view/Xt3SzX
float iCapsule( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                in vec3 pa, in vec3 pb, in float r ) {
    vec3  ba = pb - pa;
    vec3  oa = ro - pa;

    float baba = dot(ba,ba);
    float bard = dot(ba,rd);
    float baoa = dot(ba,oa);
    float rdoa = dot(rd,oa);
    float oaoa = dot(oa,oa);

    float a = baba      - bard*bard;
    float b = baba*rdoa - baoa*bard;
    float c = baba*oaoa - baoa*baoa - r*r*baba;
    float h = b*b - a*c;
    if (h >= 0.) {
        float t = (-b-sqrt(h))/a;
        float d = max_dist;
        
        float y = baoa + t*bard;
        
        // body
        if (y > 0. && y < baba) {
            d = t;
        } else {
            // caps
            vec3 oc = (y <= 0.) ? oa : ro - pb;
            b = dot(rd,oc);
            c = dot(oc,oc) - r*r;
            h = b*b - c;
            if( h>0.0 ) {
                d = -b - sqrt(h);
            }
        }
        if (d >= distBound.x && d <= distBound.y) {
            vec3  pa = ro + rd * d - pa;
            float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
            normal = (pa - h*ba)/r;
            return d;
        }
    }
    return max_dist;
}

// Capped Cone:     https://www.shadertoy.com/view/llcfRf
float iCone( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
             in vec3  pa, in vec3  pb, in float ra, in float rb ) {
    vec3  ba = pb - pa;
    vec3  oa = ro - pa;
    vec3  ob = ro - pb;
    
    float m0 = dot(ba,ba);
    float m1 = dot(oa,ba);
    float m2 = dot(ob,ba); 
    float m3 = dot(rd,ba);

    //caps
    if (m1 < 0.) { 
        if( dot2(oa*m3-rd*m1)<(ra*ra*m3*m3) ) {
            float d = -m1/m3;
            if (d >= distBound.x && d <= distBound.y) {
                normal = -ba*inversesqrt(m0);
                return d;
            }
        }
    }
    else if (m2 > 0.) { 
        if( dot2(ob*m3-rd*m2)<(rb*rb*m3*m3) ) {
            float d = -m2/m3;
            if (d >= distBound.x && d <= distBound.y) {
                normal = ba*inversesqrt(m0);
                return d;
            }
        }
    }
                       
    // body
    float m4 = dot(rd,oa);
    float m5 = dot(oa,oa);
    float rr = ra - rb;
    float hy = m0 + rr*rr;
    
    float k2 = m0*m0    - m3*m3*hy;
    float k1 = m0*m0*m4 - m1*m3*hy + m0*ra*(rr*m3*1.0        );
    float k0 = m0*m0*m5 - m1*m1*hy + m0*ra*(rr*m1*2.0 - m0*ra);
    
    float h = k1*k1 - k2*k0;
    if( h < 0. ) return max_dist;

    float t = (-k1-sqrt(h))/k2;

    float y = m1 + t*m3;
    if (y > 0. && y < m0 && t >= distBound.x && t <= distBound.y) {
        normal = normalize(m0*(m0*(oa+t*rd)+rr*ba*ra)-ba*hy*y);
        return t;
    } else {   
	    return max_dist;
    }
}

// Ellipsoid:       https://www.shadertoy.com/view/MlsSzn
float iEllipsoid( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                  in vec3 rad ) {
    vec3 ocn = ro / rad;
    vec3 rdn = rd / rad;
    
    float a = dot( rdn, rdn );
	float b = dot( ocn, rdn );
	float c = dot( ocn, ocn );
	float h = b*b - a*(c-1.);
    
    if (h < 0.) {
        return max_dist;
    }
    
	float d = (-b - sqrt(h))/a;
    
    if (d < distBound.x || d > distBound.y) {
        return max_dist;
    } else {
        normal = normalize((ro + d*rd)/rad);
    	return d;
    }
}

// Rounded Cone:    https://www.shadertoy.com/view/MlKfzm
float iRoundedCone( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                    in vec3  pa, in vec3  pb, in float ra, in float rb ) {
    vec3  ba = pb - pa;
	vec3  oa = ro - pa;
	vec3  ob = ro - pb;
    float rr = ra - rb;
    float m0 = dot(ba,ba);
    float m1 = dot(ba,oa);
    float m2 = dot(ba,rd);
    float m3 = dot(rd,oa);
    float m5 = dot(oa,oa);
	float m6 = dot(ob,rd);
    float m7 = dot(ob,ob);
    
    float d2 = m0-rr*rr;
    
	float k2 = d2    - m2*m2;
    float k1 = d2*m3 - m1*m2 + m2*rr*ra;
    float k0 = d2*m5 - m1*m1 + m1*rr*ra*2. - m0*ra*ra;
    
	float h = k1*k1 - k0*k2;
    if (h < 0.0) {
        return max_dist;
    }
    
    float t = (-sqrt(h)-k1)/k2;
    
    float y = m1 - ra*rr + t*m2;
    if (y>0.0 && y<d2) {
        if (t >= distBound.x && t <= distBound.y) {
        	normal = normalize( d2*(oa + t*rd)-ba*y );
            return t;
        } else {
            return max_dist;
        }
    } else {
        float h1 = m3*m3 - m5 + ra*ra;
        float h2 = m6*m6 - m7 + rb*rb;

        if (max(h1,h2)<0.0) {
            return max_dist;
        }

        vec3 n = vec3(0);
        float r = max_dist;

        if (h1 > 0.) {        
            r = -m3 - sqrt( h1 );
            n = (oa+r*rd)/ra;
        }
        if (h2 > 0.) {
            t = -m6 - sqrt( h2 );
            if( t<r ) {
                n = (ob+t*rd)/rb;
                r = t;
            }
        }
        if (r >= distBound.x && r <= distBound.y) {
            normal = n;
            return r;
        } else {
            return max_dist;
        }
    }
}

// Triangle:        https://www.shadertoy.com/view/MlGcDz
float iTriangle( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                 in vec3 v0, in vec3 v1, in vec3 v2 ) {
    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = ro - v0;

    vec3  n = cross( v1v0, v2v0 );
    vec3  q = cross( rov0, rd );
    float d = 1.0/dot( rd, n );
    float u = d*dot( -q, v2v0 );
    float v = d*dot(  q, v1v0 );
    float t = d*dot( -n, rov0 );

    if( u<0. || v<0. || (u+v)>1. || t<distBound.x || t>distBound.y) {
        return max_dist;
    } else {
        normal = normalize(-n);
        return t;
    }
}

// Sphere4:         https://www.shadertoy.com/view/3tj3DW
float iSphere4( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                in float ra ) {
    // -----------------------------
    // solve quartic equation
    // -----------------------------
    
    float r2 = ra*ra;
    
    vec3 d2 = rd*rd; vec3 d3 = d2*rd;
    vec3 o2 = ro*ro; vec3 o3 = o2*ro;

    float ka = 1.0/dot(d2,d2);

    float k0 = ka* dot(ro,d3);
    float k1 = ka* dot(o2,d2);
    float k2 = ka* dot(o3,rd);
    float k3 = ka*(dot(o2,o2) - r2*r2);

    // -----------------------------
    // solve cubic
    // -----------------------------

    float c0 = k1 - k0*k0;
    float c1 = k2 + 2.0*k0*(k0*k0 - (3.0/2.0)*k1);
    float c2 = k3 - 3.0*k0*(k0*(k0*k0 - 2.0*k1) + (4.0/3.0)*k2);

    float p = c0*c0*3.0 + c2;
    float q = c0*c0*c0 - c0*c2 + c1*c1;
    float h = q*q - p*p*p*(1.0/27.0);

    // -----------------------------
    // skip the case of 3 real solutions for the cubic, which involves 
    // 4 complex solutions for the quartic, since we know this objcet is 
    // convex
    // -----------------------------
    if (h<0.0) {
        return max_dist;
    }
    
    // one real solution, two complex (conjugated)
    h = sqrt(h);

    float s = sign(q+h)*pow(abs(q+h),1.0/3.0); // cuberoot
    float t = sign(q-h)*pow(abs(q-h),1.0/3.0); // cuberoot

    vec2 v = vec2( (s+t)+c0*4.0, (s-t)*sqrt(3.0) )*0.5;
    
    // -----------------------------
    // the quartic will have two real solutions and two complex solutions.
    // we only want the real ones
    // -----------------------------
    
    float r = length(v);
	float d = -abs(v.y)/sqrt(r+v.x) - c1/r - k0;

    if (d >= distBound.x && d <= distBound.y) {
	    vec3 pos = ro + rd * d;
	    normal = normalize( pos*pos*pos );
	    return d;
    } else {
        return max_dist;
    }
}

// Goursat:         https://www.shadertoy.com/view/3lj3DW

float cuberoot( float x ) { return sign(x)*pow(abs(x),1.0/3.0); }

float iGoursat( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
                in float ra, float rb ) {
// hole: x4 + y4 + z4 - (r2^2)·(x2 + y2 + z2) + r1^4 = 0;
    float ra2 = ra*ra;
    float rb2 = rb*rb;
    
    vec3 rd2 = rd*rd; vec3 rd3 = rd2*rd;
    vec3 ro2 = ro*ro; vec3 ro3 = ro2*ro;

    float ka = 1.0/dot(rd2,rd2);

    float k3 = ka*(dot(ro ,rd3));
    float k2 = ka*(dot(ro2,rd2) - rb2/6.0);
    float k1 = ka*(dot(ro3,rd ) - rb2*dot(rd,ro)/2.0  );
    float k0 = ka*(dot(ro2,ro2) + ra2*ra2 - rb2*dot(ro,ro) );

    float c2 = k2 - k3*(k3);
    float c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
    float c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);

    c0 /= 3.0;

    float Q = c2*c2 + c0;
    float R = c2*c2*c2 - 3.0*c0*c2 + c1*c1;
    float h = R*R - Q*Q*Q;
    
    
    // 2 intersections
    if (h>0.0) {
        h = sqrt(h);

        float s = cuberoot( R + h );
        float u = cuberoot( R - h );
        
        float x = s+u+4.0*c2;
        float y = s-u;
        
        float k2 = x*x + y*y*3.0;
  
        float k = sqrt(k2);

		float d = -0.5*abs(y)*sqrt(6.0/(k+x)) 
                  -2.0*c1*(k+x)/(k2+x*k) 
                  -k3;
        
        if (d >= distBound.x && d <= distBound.y) {
            vec3 pos = ro + rd * d;
            normal = normalize( 4.0*pos*pos*pos - 2.0*pos*rb*rb );
            return d;
        } else {
            return max_dist;
        }
    } else {	
        // 4 intersections
        float sQ = sqrt(Q);
        float z = c2 - 2.0*sQ*cos( acos(-R/(sQ*Q)) / 3.0 );

        float d1 = z   - 3.0*c2;
        float d2 = z*z - 3.0*c0;

        if (abs(d1)<1.0e-4) {  
            if( d2<0.0) return max_dist;
            d2 = sqrt(d2);
        } else {
            if (d1<0.0) return max_dist;
            d1 = sqrt( d1/2.0 );
            d2 = c1/d1;
        }

        //----------------------------------

        float h1 = sqrt(d1*d1 - z + d2);
        float h2 = sqrt(d1*d1 - z - d2);
        float t1 = -d1 - h1 - k3;
        float t2 = -d1 + h1 - k3;
        float t3 =  d1 - h2 - k3;
        float t4 =  d1 + h2 - k3;

        if (t2<0.0 && t4<0.0) return max_dist;

        float result = 1e20;
             if (t1>0.0) result=t1;
        else if (t2>0.0) result=t2;
             if (t3>0.0) result=min(result,t3);
        else if (t4>0.0) result=min(result,t4);

        if (result >= distBound.x && result <= distBound.y) {
            vec3 pos = ro + rd * result;
            normal = normalize( 4.0*pos*pos*pos - 2.0*pos*rb*rb );
            return result;
        } else {
            return max_dist;
        }
    }
}

// Rounded Box:     https://www.shadertoy.com/view/WlSXRW
float iRoundedBox(in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
   				  in vec3 size, in float rad ) {
	// bounding box
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*(size+rad);
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
	float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
    if (tN > tF || tF < 0.0) {
    	return max_dist;
    }
    float t = (tN>=distBound.x&&tN<=distBound.y)?tN:
    		  (tF>=distBound.x&&tF<=distBound.y)?tF:max_dist;

    // convert to first octant
    vec3 pos = ro+t*rd;
    vec3 s = sign(pos);
    vec3 ros = ro*s;
    vec3 rds = rd*s;
    pos *= s;
        
    // faces
    pos -= size;
    pos = max( pos.xyz, pos.yzx );
    if (min(min(pos.x,pos.y),pos.z)<0.0) {
        if (t >= distBound.x && t <= distBound.y) {
            vec3 p = ro + rd * t;
            normal = sign(p)*normalize(max(abs(p)-size,0.0));
            return t;
        }
    }
    
    // some precomputation
    vec3 oc = ros - size;
    vec3 dd = rds*rds;
	vec3 oo = oc*oc;
    vec3 od = oc*rds;
    float ra2 = rad*rad;

    t = max_dist;        

    // corner
    {
    float b = od.x + od.y + od.z;
	float c = oo.x + oo.y + oo.z - ra2;
	float h = b*b - c;
	if (h > 0.0) t = -b-sqrt(h);
    }

    // edge X
    {
	float a = dd.y + dd.z;
	float b = od.y + od.z;
	float c = oo.y + oo.z - ra2;
	float h = b*b - a*c;
	if (h>0.0) {
	  h = (-b-sqrt(h))/a;
      if (h>=distBound.x && h<t && abs(ros.x+rds.x*h)<size.x ) t = h;
    }
	}
    // edge Y
    {
	float a = dd.z + dd.x;
	float b = od.z + od.x;
	float c = oo.z + oo.x - ra2;
	float h = b*b - a*c;
	if (h>0.0) {
	  h = (-b-sqrt(h))/a;
      if (h>=distBound.x && h<t && abs(ros.y+rds.y*h)<size.y) t = h;
    }
	}
    // edge Z
    {
	float a = dd.x + dd.y;
	float b = od.x + od.y;
	float c = oo.x + oo.y - ra2;
	float h = b*b - a*c;
	if (h>0.0) {
	  h = (-b-sqrt(h))/a;
      if (h>=distBound.x && h<t && abs(ros.z+rds.z*h)<size.z) t = h;
    }
	}
    
	if (t >= distBound.x && t <= distBound.y) {
        vec3 p = ro + rd * t;
        normal = sign(p)*normalize(max(abs(p)-size,1e-16));
        return t;
    } else {
        return max_dist;
    };
}





float raw_ds(vec3 p) {
  float d = max_dist;

  d = min(d, sphere(p - sphere_center, 1.));
  d = min(d, plane(p - plane_center, -normalize(vec3(0., 1., 0.))));
  d = min(d, cylinder(p - cylinder_center, vec3(0., 1., 0.), 1.));

  d = min(d, triangle(p - vec3(2., 2., 2.), triangle_pts[0], triangle_pts[1], triangle_pts[2]));

  return d;
}

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

vec3 dist_scene_gradient(vec3 p) {
  float d = raw_ds(p);
  vec2 e = vec2(min_dist, 0);

  return (vec3(raw_ds(p + e.xyy), raw_ds(p + e.yxy), raw_ds(p + e.yyx)) - d) / min_dist;
}

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






vec2 sample_circle(in float t) {
  float phi = t * TWO_PI;
  return vec2(cos(phi), sin(phi));
}
vec2 sample_incircle(in vec2 t) {
  return sample_circle(t.x) * sqrt(t.y);
}
vec3 sample_cosine_weighted_sphere(in vec2 t) {
  float phi = TWO_PI * t.y;
  float sin_theta = sqrt(1. - t.x);
  return vec3(sin_theta * cos(phi), sin_theta * sin(phi), sqrt(t.x)); 
}
vec3 sample_sphere(in vec2 t) {
  vec2 uv = vec2(t.x * 2. - 1., t.y);
  float sinTheta = sqrt(1 - uv.x * uv.x); 
  float phi = TWO_PI * uv.y; 
  float x = sinTheta * cos(phi); 
  float z = sinTheta * sin(phi); 
  return vec3(x, uv.x, z); 
}
vec3 sample_insphere(in vec3 t) {
  return sample_sphere(t.xy) * cbrt(t.z); 
}
vec2 sample_insquare(in vec2 t) {
  return (2. * t - 1.); 
}

float in_shadow(in Ray ray, in float mag_sq) {
  Hit hitObj = scene(ray);
  bool hit = hitObj.hit;
  float ds = hitObj.dist;

  return !hit || ds * ds >= mag_sq ? 1. : 0.;
}

const float k = 1./(2.*TWO_PI);

Light sample_light(in vec2 _p) {
  vec2 p = 2. * sample_insquare(_p);
  return Light(light_pos + vec3(p.x, 0., p.y), 4. * light_color);
}

vec3 sun_light_col(in vec3 dir, in vec3 norm, in vec3 light_color) {
  return max(dot(dir, norm), 0.) * light_color;
}
vec3 sun_light_col(in vec3 dir, in vec3 norm) {
  return max(dot(dir, norm), 0.) * sun_color;
}

vec3 point_light_col(in vec3 pos, in vec3 norm, in Light light) {
  vec3 d = light.pos - pos;
  float mag_sq = dot(d, d);
  float mag = sqrt(mag_sq);
  vec3 dir = d / mag;
  return sun_light_col(dir, norm, light.color) / mag_sq;
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




float reflectance(float cos_angle, float index) {
  // Use Schlick's approximation for reflectance.
  float r0 = (1-index) / (1+index);
  r0 = r0 * r0;
  return mix(1., pow((1 - cos_angle), 5), r0);
}

float _reflectance(float cos_i, float cos_t, float index) {
  float s = (index * cos_i - cos_t) / (index * cos_i + cos_t);
  float p = (index * cos_t - cos_i) / (index * cos_t + cos_i);
  vec2 sp = vec2(s, p);

  return min(0.5 * dot(sp, sp), 1.);
}

float _reflectance(float cos_i, float index) {
  float cos_t = sqrt(1. - index * (1. - cos_i * cos_i));
  return _reflectance(cos_i, cos_t, index);
}







vec3 LinearToSRGB(vec3 rgb)
{
  rgb = clamp(rgb, 0.0f, 1.0f);
    
  return mix(
    pow(rgb, vec3(1.0 / 2.4)) * 1.055 - 0.055,
    rgb * 12.92,
    lessThan(rgb, vec3(0.0031308))
  );
}
 
vec3 SRGBToLinear(vec3 rgb)
{
  rgb = clamp(rgb, 0.0f, 1.0f);
    
  return mix(
    pow(((rgb + 0.055) / 1.055), vec3(2.4)),
    rgb / 12.92,
    lessThan(rgb, vec3(0.04045))
  );
}

vec3 ACESFilm(vec3 x)
{
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0f, 1.0f);
}

float sq(float v) {
  return v * v;
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

    for (int i = 0; i < int(gi_reflection_depth) + 1; ++i) {
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
      float scatter_t = -log(random_0t1(uv_seed, x)) * sigma_t;

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
          //     if (scatter_t < t && i < int(gi_reflection_depth)) {
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
            dir = normalize(sample_sphere(random_0t1_2(uv_seed, x)) + ray.dir * sigma_f);
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