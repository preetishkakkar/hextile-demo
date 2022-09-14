
//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
#include "std_cbuffer.h"


//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float3 Position     : POSITION;
};

struct VS_OUTPUT
{
    float4 Position     : SV_POSITION;
};

VS_OUTPUT RenderSceneVS( VS_INPUT input)
{
	VS_OUTPUT Output;

	float3 vP = mul( float4(input.Position.xyz,1.0), g_mLocToWorld ).xyz;
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul( float4(vP,1.0), g_mViewProjection );

	return Output;    
}


float4 WhitePS( VS_OUTPUT In ) : SV_TARGET0
{
	return float4(1,1,1,1);
}
