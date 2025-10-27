static inline void mpeg_motion(MpegEncContext *s,

                               UINT8 *dest_y, UINT8 *dest_cb, UINT8 *dest_cr,

                               int dest_offset,

                               UINT8 **ref_picture, int src_offset,

                               int field_based, op_pixels_func *pix_op,

                               int motion_x, int motion_y, int h)

{

    UINT8 *ptr;

    int dxy, offset, mx, my, src_x, src_y, height, linesize;

if(s->quarter_sample)

{

    motion_x>>=1;

    motion_y>>=1;

}



    dxy = ((motion_y & 1) << 1) | (motion_x & 1);

    src_x = s->mb_x * 16 + (motion_x >> 1);

    src_y = s->mb_y * (16 >> field_based) + (motion_y >> 1);

                

    /* WARNING: do no forget half pels */

    height = s->height >> field_based;

    src_x = clip(src_x, -16, s->width);

    if (src_x == s->width)

        dxy &= ~1;

    src_y = clip(src_y, -16, height);

    if (src_y == height)

        dxy &= ~2;

    linesize = s->linesize << field_based;

    ptr = ref_picture[0] + (src_y * linesize) + (src_x) + src_offset;

    dest_y += dest_offset;

    pix_op[dxy](dest_y, ptr, linesize, h);

    pix_op[dxy](dest_y + 8, ptr + 8, linesize, h);



    if (s->out_format == FMT_H263) {

        dxy = 0;

        if ((motion_x & 3) != 0)

            dxy |= 1;

        if ((motion_y & 3) != 0)

            dxy |= 2;

        mx = motion_x >> 2;

        my = motion_y >> 2;

    } else {

        mx = motion_x / 2;

        my = motion_y / 2;

        dxy = ((my & 1) << 1) | (mx & 1);

        mx >>= 1;

        my >>= 1;

    }

    

    src_x = s->mb_x * 8 + mx;

    src_y = s->mb_y * (8 >> field_based) + my;

    src_x = clip(src_x, -8, s->width >> 1);

    if (src_x == (s->width >> 1))

        dxy &= ~1;

    src_y = clip(src_y, -8, height >> 1);

    if (src_y == (height >> 1))

        dxy &= ~2;



    offset = (src_y * (linesize >> 1)) + src_x + (src_offset >> 1);

    ptr = ref_picture[1] + offset;

    pix_op[dxy](dest_cb + (dest_offset >> 1), ptr, linesize >> 1, h >> 1);

    ptr = ref_picture[2] + offset;

    pix_op[dxy](dest_cr + (dest_offset >> 1), ptr, linesize >> 1, h >> 1);

}
