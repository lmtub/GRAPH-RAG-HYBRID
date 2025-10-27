static int ivi_mc(ivi_mc_func mc, int16_t *buf, const int16_t *ref_buf,

                  int offs, int mv_x, int mv_y, uint32_t pitch,

                  int mc_type)

{

    int ref_offs = offs + mv_y * pitch + mv_x;



    if (offs < 0 || ref_offs < 0 || !ref_buf)

        return AVERROR_INVALIDDATA;



    mc(buf + offs, ref_buf + ref_offs, pitch, mc_type);



    return 0;

}
