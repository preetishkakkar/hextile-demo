// HLSL function stub generated
#ifndef M_PI
	#define M_PI 3.1415926535897932384626433832795
#endif

static float g_fallOffContrast = 0.6;
static float g_exp = 7;

SamplerState g_samWrap : register(s0); // can't disambiguate
Texture2D<float4> g_trx_d : register(t0);


#include "std_cbuffer.h"

struct VS_INPUT
{
	float3 Position     : POSITION;
};

struct VS_OUTPUT
{
    float4 gl_Position   : SV_POSITION;
	float3 v_position	: TEXCOORD1;
};

float3 Gain3(float3 x, float r)
{
	// increase contrast when r>0.5 and
	// reduce contrast if less
	float k = log(1-r) / log(0.5);

	float3 s = 2*step(0.5, x);
	float3 m = 2*(1 - s);

	float3 res = 0.5*s + 0.25*m * pow(max(0.0, s + x*m), k);
	
	return res.xyz / (res.x+res.y+res.z);
}

float2x2 LoadRot2x2(int2 idx, float rotStrength)
{
	float angle = abs(idx.x*idx.y) + abs(idx.x+idx.y) + M_PI;

	// remap to +/-pi
	angle = fmod(angle, 2*M_PI); 
	if(angle<0) angle += 2*M_PI;
	if(angle>M_PI) angle -= 2*M_PI;

	angle *= rotStrength;

	float cs = cos(angle), si = sin(angle);

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
	// Scaling of the input
	st *= 2 * sqrt(3);

	// Skew input space into simplex triangle grid
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

void hex2colTex(out float4 color, out float3 weights,
				Texture2D tex, SamplerState samp, float2 st,
				float rotStrength, float r=0.5)
{
	float2 dSTdx = ddx(st), dSTdy = ddy(st);

	// Get triangle info
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

	// Fetch input
	float4 c1 = tex.SampleGrad(samp, st1, 
							   mul(dSTdx, rot1), mul(dSTdy, rot1));
	float4 c2 = tex.SampleGrad(samp, st2,
							   mul(dSTdx, rot2), mul(dSTdy, rot2));
	float4 c3 = tex.SampleGrad(samp, st3, 
							   mul(dSTdx, rot3), mul(dSTdy, rot3));

	// use luminance as weight
	float3 Lw = float3(0.299, 0.587, 0.114);
	float3 Dw = float3(dot(c1.xyz,Lw),dot(c2.xyz,Lw),dot(c3.xyz,Lw));
	
	Dw = lerp(1.0, Dw, g_fallOffContrast);	// 0.6
	float3 W = Dw*pow(float3(w1, w2, w3), g_exp);	// 7
	W /= (W.x+W.y+W.z);
	if(r!=0.5) W = Gain3(W, r);

	color = W.x * c1 + W.y * c2 + W.z * c3;
	weights = ProduceHexWeights(W.xyz, vertex1, vertex2, vertex3);
}

static float TILE_RATE = 5.0;

float GetTileRate()
{
	return 0.05*TILE_RATE;
}

void FetchColorAndWeight(out float3 color, out float3 weights, float2 st)
{
	float4 col4;
	hex2colTex(col4, weights, g_trx_d, g_samWrap, st, 0, 0.7);
	color = col4.xyz;
}

struct OutputStruct {
	float4 Target0 : SV_Target0;
};

VS_OUTPUT RenderSceneVS(float3 a_position : POSITION)
{
	VS_OUTPUT Output;

	float3 vP = a_position.xyz;// mul(float4(a_position.xyz, 1.0), g_mLocToWorld).xyz;

	// Transform the position from object space to homogeneous projection space
	Output.gl_Position = mul(float4(vP, 1.0), g_mViewProjection);
	Output.v_position.xyz = vP;

	return Output;
}


float3 Prologue(VS_OUTPUT In)
{
	return In.v_position.xyz;
	//return In.normal;
	// position in camera space
	//float4 v4ScrPos = float4(In.Position.xyz, 1);



	////float4 winInv = mul(v4ScrPos, g_mWindowInv);
	//float4 winInv;
	//winInv.x = ((v4ScrPos.x / 1280.0) * 2.0) - 1;
	//winInv.y = 1 - ((v4ScrPos.y / 960.0) * 2.0);
	//winInv.z = v4ScrPos.z;
	//winInv.w = 1.0;

	//float4 v4ViewPos = mul(winInv, g_mProjInv);
	//float3 surfPosInView = v4ViewPos.xyz / v4ViewPos.w;

	//// actual world space position
	//float3 surfPosInWorld = mul(float4(surfPosInView.xyz,1.0), g_mViewToWorld).xyz; // g_mViewToWorld is just viewMat

	//if (In.Color.x == surfPosInWorld.x) 
	//{
	//	return In.Color.xyz;
	//}

	//return surfPosInWorld;
}

float4 GroundExamplePS( VS_OUTPUT In ) : SV_TARGET0
{
	float3 sp = GetTileRate() * In.v_position.xyz;

	float2 st0 = float2(sp.x, -sp.z);	// since looking at -Z in a right hand coordinate frame.

	// need to negate .y of derivative due to upper-left corner being the texture origin
	st0 = float2(st0.x, 1.0-st0.y);

	float3 color, weights;
	FetchColorAndWeight(color, weights, st0);

	return float4(color.xyz, 1.);
}
