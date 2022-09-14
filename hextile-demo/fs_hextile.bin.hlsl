// shaderc command line:
// shadercDebug.exe -f E:\PersonalProjs\bgfx\examples\48-hextile\fs_hextile.sc -o E:\PersonalProjs\bgfx\examples\48-hextile\fs_hextile.bin --type f -i E:\PersonalProjs\bgfx\src --varyingdef E:\PersonalProjs\bgfx\examples\48-hextile\varying.def.sc --platform windows -p ps_5_0 --debug


SamplerState g_samWrap : register(s0); // can't disambiguate
Texture2D<float4> g_trx_d : register(t0);

float intBitsToFloat(int _x) { return asfloat(_x); }
float2 intBitsToFloat(uint2 _x) { return asfloat(_x); }
float3 intBitsToFloat(uint3 _x) { return asfloat(_x); }
float4 intBitsToFloat(uint4 _x) { return asfloat(_x); }
float uintBitsToFloat(uint _x) { return asfloat(_x); }
float2 uintBitsToFloat(uint2 _x) { return asfloat(_x); }
float3 uintBitsToFloat(uint3 _x) { return asfloat(_x); }
float4 uintBitsToFloat(uint4 _x) { return asfloat(_x); }
uint floatBitsToUint(float _x) { return asuint(_x); }
uint2 floatBitsToUint(float2 _x) { return asuint(_x); }
uint3 floatBitsToUint(float3 _x) { return asuint(_x); }
uint4 floatBitsToUint(float4 _x) { return asuint(_x); }
int floatBitsToInt(float _x) { return asint(_x); }
int2 floatBitsToInt(float2 _x) { return asint(_x); }
int3 floatBitsToInt(float3 _x) { return asint(_x); }
int4 floatBitsToInt(float4 _x) { return asint(_x); }
uint bitfieldReverse(uint _x) { return reversebits(_x); }
uint2 bitfieldReverse(uint2 _x) { return reversebits(_x); }
uint3 bitfieldReverse(uint3 _x) { return reversebits(_x); }
uint4 bitfieldReverse(uint4 _x) { return reversebits(_x); }
uint packHalf2x16(float2 _x)
{
return (f32tof16(_x.y)<<16) | f32tof16(_x.x);
}
float2 unpackHalf2x16(uint _x)
{
return float2(f16tof32(_x & 0xffff), f16tof32(_x >> 16) );
}
struct BgfxSampler2D
{
SamplerState m_sampler;
Texture2D m_texture;
};
struct BgfxISampler2D
{
Texture2D<int4> m_texture;
};
struct BgfxUSampler2D
{
Texture2D<uint4> m_texture;
};
struct BgfxSampler2DArray
{
SamplerState m_sampler;
Texture2DArray m_texture;
};
struct BgfxSampler2DShadow
{
SamplerComparisonState m_sampler;
Texture2D m_texture;
};
struct BgfxSampler2DArrayShadow
{
SamplerComparisonState m_sampler;
Texture2DArray m_texture;
};
struct BgfxSampler3D
{
SamplerState m_sampler;
Texture3D m_texture;
};
struct BgfxISampler3D
{
Texture3D<int4> m_texture;
};
struct BgfxUSampler3D
{
Texture3D<uint4> m_texture;
};
struct BgfxSamplerCube
{
SamplerState m_sampler;
TextureCube m_texture;
};
struct BgfxSamplerCubeShadow
{
SamplerComparisonState m_sampler;
TextureCube m_texture;
};
struct BgfxSampler2DMS
{
Texture2DMS<float4> m_texture;
};
float4 bgfxTexture2D(BgfxSampler2D _sampler, float2 _coord)
{
return _sampler.m_texture.Sample(_sampler.m_sampler, _coord);
}
float4 bgfxTexture2DBias(BgfxSampler2D _sampler, float2 _coord, float _bias)
{
return _sampler.m_texture.SampleBias(_sampler.m_sampler, _coord, _bias);
}
float4 bgfxTexture2DLod(BgfxSampler2D _sampler, float2 _coord, float _level)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _level);
}
float4 bgfxTexture2DLodOffset(BgfxSampler2D _sampler, float2 _coord, float _level, int2 _offset)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _level, _offset);
}
float4 bgfxTexture2DProj(BgfxSampler2D _sampler, float3 _coord)
{
float2 coord = _coord.xy * rcp(_coord.z);
return _sampler.m_texture.Sample(_sampler.m_sampler, coord);
}
float4 bgfxTexture2DProj(BgfxSampler2D _sampler, float4 _coord)
{
float2 coord = _coord.xy * rcp(_coord.w);
return _sampler.m_texture.Sample(_sampler.m_sampler, coord);
}
float4 bgfxTexture2DGrad(BgfxSampler2D _sampler, float2 _coord, float2 _dPdx, float2 _dPdy)
{
return _sampler.m_texture.SampleGrad(_sampler.m_sampler, _coord, _dPdx, _dPdy);
}
float4 bgfxTexture2DArray(BgfxSampler2DArray _sampler, float3 _coord)
{
return _sampler.m_texture.Sample(_sampler.m_sampler, _coord);
}
float4 bgfxTexture2DArrayLod(BgfxSampler2DArray _sampler, float3 _coord, float _lod)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _lod);
}
float4 bgfxTexture2DArrayLodOffset(BgfxSampler2DArray _sampler, float3 _coord, float _level, int2 _offset)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _level, _offset);
}
float bgfxShadow2D(BgfxSampler2DShadow _sampler, float3 _coord)
{
return _sampler.m_texture.SampleCmpLevelZero(_sampler.m_sampler, _coord.xy, _coord.z);
}
float bgfxShadow2DProj(BgfxSampler2DShadow _sampler, float4 _coord)
{
float3 coord = _coord.xyz * rcp(_coord.w);
return _sampler.m_texture.SampleCmpLevelZero(_sampler.m_sampler, coord.xy, coord.z);
}
float4 bgfxShadow2DArray(BgfxSampler2DArrayShadow _sampler, float4 _coord)
{
return _sampler.m_texture.SampleCmpLevelZero(_sampler.m_sampler, _coord.xyz, _coord.w);
}
float4 bgfxTexture3D(BgfxSampler3D _sampler, float3 _coord)
{
return _sampler.m_texture.Sample(_sampler.m_sampler, _coord);
}
float4 bgfxTexture3DLod(BgfxSampler3D _sampler, float3 _coord, float _level)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _level);
}
int4 bgfxTexture3D(BgfxISampler3D _sampler, float3 _coord)
{
uint3 size;
_sampler.m_texture.GetDimensions(size.x, size.y, size.z);
return _sampler.m_texture.Load(int4(_coord * size, 0) );
}
uint4 bgfxTexture3D(BgfxUSampler3D _sampler, float3 _coord)
{
uint3 size;
_sampler.m_texture.GetDimensions(size.x, size.y, size.z);
return _sampler.m_texture.Load(int4(_coord * size, 0) );
}
float4 bgfxTextureCube(BgfxSamplerCube _sampler, float3 _coord)
{
return _sampler.m_texture.Sample(_sampler.m_sampler, _coord);
}
float4 bgfxTextureCubeBias(BgfxSamplerCube _sampler, float3 _coord, float _bias)
{
return _sampler.m_texture.SampleBias(_sampler.m_sampler, _coord, _bias);
}
float4 bgfxTextureCubeLod(BgfxSamplerCube _sampler, float3 _coord, float _level)
{
return _sampler.m_texture.SampleLevel(_sampler.m_sampler, _coord, _level);
}
float bgfxShadowCube(BgfxSamplerCubeShadow _sampler, float4 _coord)
{
return _sampler.m_texture.SampleCmpLevelZero(_sampler.m_sampler, _coord.xyz, _coord.w);
}
float4 bgfxTexelFetch(BgfxSampler2D _sampler, int2 _coord, int _lod)
{
return _sampler.m_texture.Load(int3(_coord, _lod) );
}
float4 bgfxTexelFetchOffset(BgfxSampler2D _sampler, int2 _coord, int _lod, int2 _offset)
{
return _sampler.m_texture.Load(int3(_coord, _lod), _offset );
}
float2 bgfxTextureSize(BgfxSampler2D _sampler, int _lod)
{
float2 result;
_sampler.m_texture.GetDimensions(result.x, result.y);
return result;
}
float2 bgfxTextureSize(BgfxISampler2D _sampler, int _lod)
{
float2 result;
_sampler.m_texture.GetDimensions(result.x, result.y);
return result;
}
float2 bgfxTextureSize(BgfxUSampler2D _sampler, int _lod)
{
float2 result;
_sampler.m_texture.GetDimensions(result.x, result.y);
return result;
}
float4 bgfxTextureGather0(BgfxSampler2D _sampler, float2 _coord)
{
return _sampler.m_texture.GatherRed(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather1(BgfxSampler2D _sampler, float2 _coord)
{
return _sampler.m_texture.GatherGreen(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather2(BgfxSampler2D _sampler, float2 _coord)
{
return _sampler.m_texture.GatherBlue(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather3(BgfxSampler2D _sampler, float2 _coord)
{
return _sampler.m_texture.GatherAlpha(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGatherOffset0(BgfxSampler2D _sampler, float2 _coord, int2 _offset)
{
return _sampler.m_texture.GatherRed(_sampler.m_sampler, _coord, _offset);
}
float4 bgfxTextureGatherOffset1(BgfxSampler2D _sampler, float2 _coord, int2 _offset)
{
return _sampler.m_texture.GatherGreen(_sampler.m_sampler, _coord, _offset);
}
float4 bgfxTextureGatherOffset2(BgfxSampler2D _sampler, float2 _coord, int2 _offset)
{
return _sampler.m_texture.GatherBlue(_sampler.m_sampler, _coord, _offset);
}
float4 bgfxTextureGatherOffset3(BgfxSampler2D _sampler, float2 _coord, int2 _offset)
{
return _sampler.m_texture.GatherAlpha(_sampler.m_sampler, _coord, _offset);
}
float4 bgfxTextureGather0(BgfxSampler2DArray _sampler, float3 _coord)
{
return _sampler.m_texture.GatherRed(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather1(BgfxSampler2DArray _sampler, float3 _coord)
{
return _sampler.m_texture.GatherGreen(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather2(BgfxSampler2DArray _sampler, float3 _coord)
{
return _sampler.m_texture.GatherBlue(_sampler.m_sampler, _coord);
}
float4 bgfxTextureGather3(BgfxSampler2DArray _sampler, float3 _coord)
{
return _sampler.m_texture.GatherAlpha(_sampler.m_sampler, _coord);
}
int4 bgfxTexelFetch(BgfxISampler2D _sampler, int2 _coord, int _lod)
{
return _sampler.m_texture.Load(int3(_coord, _lod) );
}
uint4 bgfxTexelFetch(BgfxUSampler2D _sampler, int2 _coord, int _lod)
{
return _sampler.m_texture.Load(int3(_coord, _lod) );
}
float4 bgfxTexelFetch(BgfxSampler2DMS _sampler, int2 _coord, int _sampleIdx)
{
return _sampler.m_texture.Load(_coord, _sampleIdx);
}
float4 bgfxTexelFetch(BgfxSampler2DArray _sampler, int3 _coord, int _lod)
{
return _sampler.m_texture.Load(int4(_coord, _lod) );
}
float4 bgfxTexelFetch(BgfxSampler3D _sampler, int3 _coord, int _lod)
{
return _sampler.m_texture.Load(int4(_coord, _lod) );
}
float3 bgfxTextureSize(BgfxSampler3D _sampler, int _lod)
{
float3 result;
_sampler.m_texture.GetDimensions(result.x, result.y, result.z);
return result;
}
float3 instMul(float3 _vec, float3x3 _mtx) { return mul(_mtx, _vec); }
float3 instMul(float3x3 _mtx, float3 _vec) { return mul(_vec, _mtx); }
float4 instMul(float4 _vec, float4x4 _mtx) { return mul(_mtx, _vec); }
float4 instMul(float4x4 _mtx, float4 _vec) { return mul(_vec, _mtx); }
bool2 lessThan(float2 _a, float2 _b) { return _a < _b; }
bool3 lessThan(float3 _a, float3 _b) { return _a < _b; }
bool4 lessThan(float4 _a, float4 _b) { return _a < _b; }
bool2 lessThanEqual(float2 _a, float2 _b) { return _a <= _b; }
bool3 lessThanEqual(float3 _a, float3 _b) { return _a <= _b; }
bool4 lessThanEqual(float4 _a, float4 _b) { return _a <= _b; }
bool2 greaterThan(float2 _a, float2 _b) { return _a > _b; }
bool3 greaterThan(float3 _a, float3 _b) { return _a > _b; }
bool4 greaterThan(float4 _a, float4 _b) { return _a > _b; }
bool2 greaterThanEqual(float2 _a, float2 _b) { return _a >= _b; }
bool3 greaterThanEqual(float3 _a, float3 _b) { return _a >= _b; }
bool4 greaterThanEqual(float4 _a, float4 _b) { return _a >= _b; }
bool2 notEqual(float2 _a, float2 _b) { return _a != _b; }
bool3 notEqual(float3 _a, float3 _b) { return _a != _b; }
bool4 notEqual(float4 _a, float4 _b) { return _a != _b; }
bool2 equal(float2 _a, float2 _b) { return _a == _b; }
bool3 equal(float3 _a, float3 _b) { return _a == _b; }
bool4 equal(float4 _a, float4 _b) { return _a == _b; }
float mix(float _a, float _b, float _t) { return lerp(_a, _b, _t); }
float2 mix(float2 _a, float2 _b, float2 _t) { return lerp(_a, _b, _t); }
float3 mix(float3 _a, float3 _b, float3 _t) { return lerp(_a, _b, _t); }
float4 mix(float4 _a, float4 _b, float4 _t) { return lerp(_a, _b, _t); }
float mod(float _a, float _b) { return _a - _b * floor(_a / _b); }
float2 mod(float2 _a, float2 _b) { return _a - _b * floor(_a / _b); }
float3 mod(float3 _a, float3 _b) { return _a - _b * floor(_a / _b); }
float4 mod(float4 _a, float4 _b) { return _a - _b * floor(_a / _b); }
float2 vec2_splat(float _x) { return float2(_x, _x); }
float3 vec3_splat(float _x) { return float3(_x, _x, _x); }
float4 vec4_splat(float _x) { return float4(_x, _x, _x, _x); }
uint2 uvec2_splat(uint _x) { return uint2(_x, _x); }
uint3 uvec3_splat(uint _x) { return uint3(_x, _x, _x); }
uint4 uvec4_splat(uint _x) { return uint4(_x, _x, _x, _x); }
float4x4 mtxFromRows(float4 _0, float4 _1, float4 _2, float4 _3)
{
return float4x4(_0, _1, _2, _3);
}
float4x4 mtxFromCols(float4 _0, float4 _1, float4 _2, float4 _3)
{
return transpose(float4x4(_0, _1, _2, _3) );
}
float3x3 mtxFromRows(float3 _0, float3 _1, float3 _2)
{
return float3x3(_0, _1, _2);
}
float3x3 mtxFromCols(float3 _0, float3 _1, float3 _2)
{
return transpose(float3x3(_0, _1, _2) );
}
static float4 u_viewRect;
static float4 u_viewTexel;
static float4x4 u_view;
static float4x4 u_invView;
static float4x4 u_proj;
static float4x4 u_invProj;
static float4x4 u_viewProj;
static float4x4 u_invViewProj;
static float4x4 u_model[32];
static float4x4 u_modelView;
static float4x4 u_modelViewProj;
static float4 u_alphaRef4;
float4 encodeRE8(float _r)
{
float exponent = ceil(log2(_r) );
return float4(_r / exp2(exponent)
, 0.0
, 0.0
, (exponent + 128.0) / 255.0
);
}
float decodeRE8(float4 _re8)
{
float exponent = _re8.w * 255.0 - 128.0;
return _re8.x * exp2(exponent);
}
float4 encodeRGBE8(float3 _rgb)
{
float4 rgbe8;
float maxComponent = max(max(_rgb.x, _rgb.y), _rgb.z);
float exponent = ceil(log2(maxComponent) );
rgbe8.xyz = _rgb / exp2(exponent);
rgbe8.w = (exponent + 128.0) / 255.0;
return rgbe8;
}
float3 decodeRGBE8(float4 _rgbe8)
{
float exponent = _rgbe8.w * 255.0 - 128.0;
float3 rgb = _rgbe8.xyz * exp2(exponent);
return rgb;
}
float3 encodeNormalUint(float3 _normal)
{
return _normal * 0.5 + 0.5;
}
float3 decodeNormalUint(float3 _encodedNormal)
{
return _encodedNormal * 2.0 - 1.0;
}
float2 encodeNormalSphereMap(float3 _normal)
{
return normalize(_normal.xy) * sqrt(_normal.z * 0.5 + 0.5);
}
float3 decodeNormalSphereMap(float2 _encodedNormal)
{
float zz = dot(_encodedNormal, _encodedNormal) * 2.0 - 1.0;
return float3(normalize(_encodedNormal.xy) * sqrt(1.0 - zz*zz), zz);
}
float2 octahedronWrap(float2 _val)
{
return (1.0 - abs(_val.yx) )
* mix(vec2_splat(-1.0), vec2_splat(1.0), float2(greaterThanEqual(_val.xy, vec2_splat(0.0) ) ) );
}
float2 encodeNormalOctahedron(float3 _normal)
{
_normal /= abs(_normal.x) + abs(_normal.y) + abs(_normal.z);
_normal.xy = _normal.z >= 0.0 ? _normal.xy : octahedronWrap(_normal.xy);
_normal.xy = _normal.xy * 0.5 + 0.5;
return _normal.xy;
}
float3 decodeNormalOctahedron(float2 _encodedNormal)
{
_encodedNormal = _encodedNormal * 2.0 - 1.0;
float3 normal;
normal.z = 1.0 - abs(_encodedNormal.x) - abs(_encodedNormal.y);
normal.xy = normal.z >= 0.0 ? _encodedNormal.xy : octahedronWrap(_encodedNormal.xy);
return normalize(normal);
}
float3 convertRGB2XYZ(float3 _rgb)
{
float3 xyz;
xyz.x = dot(float3(0.4124564, 0.3575761, 0.1804375), _rgb);
xyz.y = dot(float3(0.2126729, 0.7151522, 0.0721750), _rgb);
xyz.z = dot(float3(0.0193339, 0.1191920, 0.9503041), _rgb);
return xyz;
}
float3 convertXYZ2RGB(float3 _xyz)
{
float3 rgb;
rgb.x = dot(float3( 3.2404542, -1.5371385, -0.4985314), _xyz);
rgb.y = dot(float3(-0.9692660, 1.8760108, 0.0415560), _xyz);
rgb.z = dot(float3( 0.0556434, -0.2040259, 1.0572252), _xyz);
return rgb;
}
float3 convertXYZ2Yxy(float3 _xyz)
{
float inv = 1.0/dot(_xyz, float3(1.0, 1.0, 1.0) );
return float3(_xyz.y, _xyz.x*inv, _xyz.y*inv);
}
float3 convertYxy2XYZ(float3 _Yxy)
{
float3 xyz;
xyz.x = _Yxy.x*_Yxy.y/_Yxy.z;
xyz.y = _Yxy.x;
xyz.z = _Yxy.x*(1.0 - _Yxy.y - _Yxy.z)/_Yxy.z;
return xyz;
}
float3 convertRGB2Yxy(float3 _rgb)
{
return convertXYZ2Yxy(convertRGB2XYZ(_rgb) );
}
float3 convertYxy2RGB(float3 _Yxy)
{
return convertXYZ2RGB(convertYxy2XYZ(_Yxy) );
}
float3 convertRGB2Yuv(float3 _rgb)
{
float3 yuv;
yuv.x = dot(_rgb, float3(0.299, 0.587, 0.114) );
yuv.y = (_rgb.x - yuv.x)*0.713 + 0.5;
yuv.z = (_rgb.z - yuv.x)*0.564 + 0.5;
return yuv;
}
float3 convertYuv2RGB(float3 _yuv)
{
float3 rgb;
rgb.x = _yuv.x + 1.403*(_yuv.y-0.5);
rgb.y = _yuv.x - 0.344*(_yuv.y-0.5) - 0.714*(_yuv.z-0.5);
rgb.z = _yuv.x + 1.773*(_yuv.z-0.5);
return rgb;
}
float3 convertRGB2YIQ(float3 _rgb)
{
float3 yiq;
yiq.x = dot(float3(0.299, 0.587, 0.114 ), _rgb);
yiq.y = dot(float3(0.595716, -0.274453, -0.321263), _rgb);
yiq.z = dot(float3(0.211456, -0.522591, 0.311135), _rgb);
return yiq;
}
float3 convertYIQ2RGB(float3 _yiq)
{
float3 rgb;
rgb.x = dot(float3(1.0, 0.9563, 0.6210), _yiq);
rgb.y = dot(float3(1.0, -0.2721, -0.6474), _yiq);
rgb.z = dot(float3(1.0, -1.1070, 1.7046), _yiq);
return rgb;
}
float3 toLinear(float3 _rgb)
{
return pow(abs(_rgb), vec3_splat(2.2) );
}
float4 toLinear(float4 _rgba)
{
return float4(toLinear(_rgba.xyz), _rgba.w);
}
float3 toLinearAccurate(float3 _rgb)
{
float3 lo = _rgb / 12.92;
float3 hi = pow( (_rgb + 0.055) / 1.055, vec3_splat(2.4) );
float3 rgb = mix(hi, lo, float3(lessThanEqual(_rgb, vec3_splat(0.04045) ) ) );
return rgb;
}
float4 toLinearAccurate(float4 _rgba)
{
return float4(toLinearAccurate(_rgba.xyz), _rgba.w);
}
float toGamma(float _r)
{
return pow(abs(_r), 1.0/2.2);
}
float3 toGamma(float3 _rgb)
{
return pow(abs(_rgb), vec3_splat(1.0/2.2) );
}
float4 toGamma(float4 _rgba)
{
return float4(toGamma(_rgba.xyz), _rgba.w);
}
float3 toGammaAccurate(float3 _rgb)
{
float3 lo = _rgb * 12.92;
float3 hi = pow(abs(_rgb), vec3_splat(1.0/2.4) ) * 1.055 - 0.055;
float3 rgb = mix(hi, lo, float3(lessThanEqual(_rgb, vec3_splat(0.0031308) ) ) );
return rgb;
}
float4 toGammaAccurate(float4 _rgba)
{
return float4(toGammaAccurate(_rgba.xyz), _rgba.w);
}
float3 toReinhard(float3 _rgb)
{
return toGamma(_rgb/(_rgb+vec3_splat(1.0) ) );
}
float4 toReinhard(float4 _rgba)
{
return float4(toReinhard(_rgba.xyz), _rgba.w);
}
float3 toFilmic(float3 _rgb)
{
_rgb = max(vec3_splat(0.0), _rgb - 0.004);
_rgb = (_rgb*(6.2*_rgb + 0.5) ) / (_rgb*(6.2*_rgb + 1.7) + 0.06);
return _rgb;
}
float4 toFilmic(float4 _rgba)
{
return float4(toFilmic(_rgba.xyz), _rgba.w);
}
float3 toAcesFilmic(float3 _rgb)
{
float aa = 2.51f;
float bb = 0.03f;
float cc = 2.43f;
float dd = 0.59f;
float ee = 0.14f;
return saturate( (_rgb*(aa*_rgb + bb) )/(_rgb*(cc*_rgb + dd) + ee) );
}
float4 toAcesFilmic(float4 _rgba)
{
return float4(toAcesFilmic(_rgba.xyz), _rgba.w);
}
float3 luma(float3 _rgb)
{
float yy = dot(float3(0.2126729, 0.7151522, 0.0721750), _rgb);
return vec3_splat(yy);
}
float4 luma(float4 _rgba)
{
return float4(luma(_rgba.xyz), _rgba.w);
}
float3 conSatBri(float3 _rgb, float3 _csb)
{
float3 rgb = _rgb * _csb.z;
rgb = mix(luma(rgb), rgb, _csb.y);
rgb = mix(vec3_splat(0.5), rgb, _csb.x);
return rgb;
}
float4 conSatBri(float4 _rgba, float3 _csb)
{
return float4(conSatBri(_rgba.xyz, _csb), _rgba.w);
}
float3 posterize(float3 _rgb, float _numColors)
{
return floor(_rgb*_numColors) / _numColors;
}
float4 posterize(float4 _rgba, float _numColors)
{
return float4(posterize(_rgba.xyz, _numColors), _rgba.w);
}
float3 sepia(float3 _rgb)
{
float3 color;
color.x = dot(_rgb, float3(0.393, 0.769, 0.189) );
color.y = dot(_rgb, float3(0.349, 0.686, 0.168) );
color.z = dot(_rgb, float3(0.272, 0.534, 0.131) );
return color;
}
float4 sepia(float4 _rgba)
{
return float4(sepia(_rgba.xyz), _rgba.w);
}
float3 blendOverlay(float3 _base, float3 _blend)
{
float3 lt = 2.0 * _base * _blend;
float3 gte = 1.0 - 2.0 * (1.0 - _base) * (1.0 - _blend);
return mix(lt, gte, step(vec3_splat(0.5), _base) );
}
float4 blendOverlay(float4 _base, float4 _blend)
{
return float4(blendOverlay(_base.xyz, _blend.xyz), _base.w);
}
float3 adjustHue(float3 _rgb, float _hue)
{
float3 yiq = convertRGB2YIQ(_rgb);
float angle = _hue + atan2(yiq.z, yiq.y);
float len = length(yiq.yz);
return convertYIQ2RGB(float3(yiq.x, len*cos(angle), len*sin(angle) ) );
}
float4 packFloatToRgba(float _value)
{
const float4 shift = float4(256 * 256 * 256, 256 * 256, 256, 1.0);
const float4 mask = float4(0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
float4 comp = frac(_value * shift);
comp -= comp.xxyz * mask;
return comp;
}
float unpackRgbaToFloat(float4 _rgba)
{
const float4 shift = float4(1.0 / (256.0 * 256.0 * 256.0), 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0);
return dot(_rgba, shift);
}
float2 packHalfFloat(float _value)
{
const float2 shift = float2(256, 1.0);
const float2 mask = float2(0, 1.0 / 256.0);
float2 comp = frac(_value * shift);
comp -= comp.xx * mask;
return comp;
}
float unpackHalfFloat(float2 _rg)
{
const float2 shift = float2(1.0 / 256.0, 1.0);
return dot(_rg, shift);
}
float random(float2 _uv)
{
return frac(sin(dot(_uv.xy, float2(12.9898, 78.233) ) ) * 43758.5453);
}
float3 fixCubeLookup(float3 _v, float _lod, float _topLevelCubeSize)
{
float ax = abs(_v.x);
float ay = abs(_v.y);
float az = abs(_v.z);
float vmax = max(max(ax, ay), az);
float scale = 1.0 - exp2(_lod) / _topLevelCubeSize;
if (ax != vmax) { _v.x *= scale; }
if (ay != vmax) { _v.y *= scale; }
if (az != vmax) { _v.z *= scale; }
return _v;
}
float2 texture2DBc5(BgfxSampler2D _sampler, float2 _uv)
{
return bgfxTexture2D(_sampler, _uv).xy;
}
float3x3 cofactor(float4x4 _m)
{
return float3x3(
_m[1][1]*_m[2][2]-_m[1][2]*_m[2][1],
_m[1][2]*_m[2][0]-_m[1][0]*_m[2][2],
_m[1][0]*_m[2][1]-_m[1][1]*_m[2][0],
_m[0][2]*_m[2][1]-_m[0][1]*_m[2][2],
_m[0][0]*_m[2][2]-_m[0][2]*_m[2][0],
_m[0][1]*_m[2][0]-_m[0][0]*_m[2][1],
_m[0][1]*_m[1][2]-_m[0][2]*_m[1][1],
_m[0][2]*_m[1][0]-_m[0][0]*_m[1][2],
_m[0][0]*_m[1][1]-_m[0][1]*_m[1][0]
);
}
float toClipSpaceDepth(float _depthTextureZ)
{
return _depthTextureZ;
}
float3 clipToWorld(float4x4 _invViewProj, float3 _clipPos)
{
float4 wpos = mul(_invViewProj, float4(_clipPos, 1.0) );
return wpos.xyz / wpos.w;
}
uniform SamplerState s_trx_dSampler : register( s[0] ); uniform Texture2D s_trx_dTexture : register( t[0] ); static BgfxSampler2D s_trx_d = { s_trx_dSampler, s_trx_dTexture };
float3 Gain3(float3 x, float r)
{
float k = log(1-r) / log(0.5);
float3 s = 2*step(0.5, x);
float3 m = 2*(1 - s);
float3 res = 0.5*s + 0.25*m * pow(max(0.0, s + x*m), k);
return res.xyz / (res.x+res.y+res.z);
}
float2x2 LoadRot2x2(int2 idx, float rotStrength)
{
float angle = abs(idx.x*idx.y) + abs(idx.x+idx.y) + 3.1415926535897932384626433832795;
angle = fmod(angle, 2*3.1415926535897932384626433832795);
if(angle<0) angle += 2*3.1415926535897932384626433832795;
if(angle>3.1415926535897932384626433832795) angle -= 2*3.1415926535897932384626433832795;
angle *= rotStrength;
float cs = cos(angle);
float si = sin(angle);
return float2x2(cs, -si, si, cs);
}
float2 MakeCenST(int2 Vertex)
{
float2x2 invSkewMat = float2x2(1.0, 0.5, 0.0, 1.0/1.15470054);
return mul(invSkewMat, Vertex) / (2 * sqrt(3));
}
float3 ProduceHexWeights(float3 W, int2 vertex1, int2 vertex2, int2 vertex3)
{
float3 res = 0.0;
int v1 = ((vertex1.x-vertex1.y))%3;
if(v1<0) v1+=3;
int vh = v1<2 ? (v1+1) : 0;
int vl = v1>0 ? (v1-1) : 2;
int v2 = vertex1.x<vertex3.x ? vl : vh;
int v3 = vertex1.x<vertex3.x ? vh : vl;
res.x = v3==0 ? W.z : (v2==0 ? W.y : W.x);
res.y = v3==1 ? W.z : (v2==1 ? W.y : W.x);
res.z = v3==2 ? W.z : (v2==2 ? W.y : W.x);
return res;
}
float2 hash( float2 p)
{
float2 r = mul(float2x2(127.1, 311.7, 269.5, 183.3), p);
return frac( sin( r )*43758.5453 );
}
void TriangleGrid(out float w1, out float w2, out float w3,
out int2 vertex1, out int2 vertex2, out int2 vertex3,
float2 st)
{
st *= 2 * sqrt(3);
const float2x2 gridToSkewedGrid =
float2x2(1.0, -0.57735027, 0.0, 1.15470054);
float2 skewedCoord = mul(gridToSkewedGrid, st);
int2 baseId = int2( floor ( skewedCoord ));
float3 temp = float3( frac( skewedCoord ), 0);
temp.z = 1.0 - temp.x - temp.y;
float s = step(0.0, -temp.z);
float s2 = 2*s-1;
w1 = -temp.z*s2;
w2 = s - temp.y*s2;
w3 = s - temp.x*s2;
vertex1 = baseId + int2(s,s);
vertex2 = baseId + int2(s,1-s);
vertex3 = baseId + int2(1-s,s);
}
void hex2colTex(out float4 color, out float3 weights, float2 st,
float rotStrength, float r=0.5)
{
float2 dSTdx = ddx(st), dSTdy = ddy(st);
float w1, w2, w3;
int2 vertex1, vertex2, vertex3;
TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
float2 cen1 = MakeCenST(vertex1);
float2 cen2 = MakeCenST(vertex2);
float2 cen3 = MakeCenST(vertex3);
float2 st1 = mul(st - cen1, rot1) + cen1 + hash(vertex1);
float2 st2 = mul(st - cen2, rot2) + cen2 + hash(vertex2);
float2 st3 = mul(st - cen3, rot3) + cen3 + hash(vertex3);
float4 c1 = g_trx_d.SampleGrad(g_samWrap, st1, mul(dSTdx, rot1), mul(dSTdy, rot1));
float4 c2 = g_trx_d.SampleGrad(g_samWrap, st2, mul(dSTdx, rot2), mul(dSTdy, rot2));
float4 c3 = g_trx_d.SampleGrad(g_samWrap, st3, mul(dSTdx, rot3), mul(dSTdy, rot3));
float3 Lw = float3(0.299, 0.587, 0.114);
float3 Dw = float3(dot(c1.xyz,Lw),dot(c2.xyz,Lw),dot(c3.xyz,Lw));
Dw = lerp(1.0, Dw, 0.6);
float3 W = Dw*pow(float3(w1, w2, w3), 7);
W /= (W.x+W.y+W.z);
if(r!=0.5) W = Gain3(W, r);
color = W.x * c1 + W.y * c2 + W.z * c3;
weights = ProduceHexWeights(W.xyz, vertex1, vertex2, vertex3);
}
const float TILE_RATE = 5.0;
float GetTileRate()
{
return 0.05*TILE_RATE;
}
void FetchColorAndWeight(out float3 color, out float3 weights, float2 st)
{
float4 col4;
hex2colTex(col4, weights, st, 0, 0.7);
color = col4.xyz;
}
void main( float4 gl_FragCoord : SV_POSITION , float3 v_position : TEXCOORD1 , float2 v_texcoord0 : TEXCOORD0 , out float4 bgfx_FragData0 : SV_TARGET0 )
{
float4 bgfx_VoidFrag = vec4_splat(0.0);
float3 surfPosInWorld = v_position.xyz;
float3 sp = GetTileRate() * surfPosInWorld;
float2 st0 = float2(sp.x, -sp.z);
st0 = float2(st0.x, 1.0-st0.y);
float3 color, weights;
FetchColorAndWeight(color, weights, st0);
bgfx_FragData0 = float4(weights, 1.0);
}
