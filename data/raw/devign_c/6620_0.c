static inline int is_yuv_planar(PixFmtInfo *ps)

{

    return (ps->color_type == FF_COLOR_YUV ||

            ps->color_type == FF_COLOR_YUV_JPEG) && !ps->is_packed;

}
