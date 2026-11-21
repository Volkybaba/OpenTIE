#ifndef TIE_FLIGHT_MESH_TABLE_H
#define TIE_FLIGHT_MESH_TABLE_H

#include "../aeron/shaders/mesh_table_layout.hlsli"

/* One AERON_MESH_TABLE_STRIDE_VEC4-float4 record per submitted mesh
 * table. The table is a 2D RGBA32F texture (width = stride, height =
 * table count) read with texelFetch: vertex stages on GLES allow zero
 * storage blocks, so the lookup table rides a texture instead of a
 * storage buffer (same as OpenXWA scene_pbr_vertex_common.hlsli). */
Texture2D<float4> mesh_tables : register(t0, space0);

float4 flight_mesh_table_row(uint table_index, uint mesh_index, uint row)
{
    return mesh_tables.Load(int3(mesh_index * 3u + row, table_index, 0));
}

float flight_mesh_table_scalar(uint table_index, uint offset,
                               uint mesh_index)
{
    float4 packed = mesh_tables.Load(int3(offset + (mesh_index >> 2u), table_index, 0));
    return packed[mesh_index & 3u];
}

#endif
