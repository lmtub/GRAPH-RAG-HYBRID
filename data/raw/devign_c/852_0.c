static void chroma_mc(HEVCContext *s, int16_t *dst1, int16_t *dst2,

                      ptrdiff_t dststride, AVFrame *ref, const Mv *mv,

                      int x_off, int y_off, int block_w, int block_h)

{

    HEVCLocalContext *lc = &s->HEVClc;

    uint8_t *src1        = ref->data[1];

    uint8_t *src2        = ref->data[2];

    ptrdiff_t src1stride = ref->linesize[1];

    ptrdiff_t src2stride = ref->linesize[2];

    int pic_width        = s->ps.sps->width >> 1;

    int pic_height       = s->ps.sps->height >> 1;



    int mx = mv->x & 7;

    int my = mv->y & 7;



    x_off += mv->x >> 3;

    y_off += mv->y >> 3;

    src1  += y_off * src1stride + (x_off << s->ps.sps->pixel_shift);

    src2  += y_off * src2stride + (x_off << s->ps.sps->pixel_shift);



    if (x_off < EPEL_EXTRA_BEFORE || y_off < EPEL_EXTRA_AFTER ||

        x_off >= pic_width - block_w - EPEL_EXTRA_AFTER ||

        y_off >= pic_height - block_h - EPEL_EXTRA_AFTER) {

        const int edge_emu_stride = EDGE_EMU_BUFFER_STRIDE << s->ps.sps->pixel_shift;

        int offset1 = EPEL_EXTRA_BEFORE * (src1stride + (1 << s->ps.sps->pixel_shift));

        int buf_offset1 = EPEL_EXTRA_BEFORE *

                          (edge_emu_stride + (1 << s->ps.sps->pixel_shift));

        int offset2 = EPEL_EXTRA_BEFORE * (src2stride + (1 << s->ps.sps->pixel_shift));

        int buf_offset2 = EPEL_EXTRA_BEFORE *

                          (edge_emu_stride + (1 << s->ps.sps->pixel_shift));



        s->vdsp.emulated_edge_mc(lc->edge_emu_buffer, src1 - offset1,

                                 edge_emu_stride, src1stride,

                                 block_w + EPEL_EXTRA, block_h + EPEL_EXTRA,

                                 x_off - EPEL_EXTRA_BEFORE,

                                 y_off - EPEL_EXTRA_BEFORE,

                                 pic_width, pic_height);



        src1 = lc->edge_emu_buffer + buf_offset1;

        src1stride = edge_emu_stride;

        s->hevcdsp.put_hevc_epel[!!my][!!mx](dst1, dststride, src1, src1stride,

                                             block_w, block_h, mx, my, lc->mc_buffer);



        s->vdsp.emulated_edge_mc(lc->edge_emu_buffer, src2 - offset2,

                                 edge_emu_stride, src2stride,

                                 block_w + EPEL_EXTRA, block_h + EPEL_EXTRA,

                                 x_off - EPEL_EXTRA_BEFORE,

                                 y_off - EPEL_EXTRA_BEFORE,

                                 pic_width, pic_height);

        src2 = lc->edge_emu_buffer + buf_offset2;

        src2stride = edge_emu_stride;



        s->hevcdsp.put_hevc_epel[!!my][!!mx](dst2, dststride, src2, src2stride,

                                             block_w, block_h, mx, my,

                                             lc->mc_buffer);

    } else {

        s->hevcdsp.put_hevc_epel[!!my][!!mx](dst1, dststride, src1, src1stride,

                                             block_w, block_h, mx, my,

                                             lc->mc_buffer);

        s->hevcdsp.put_hevc_epel[!!my][!!mx](dst2, dststride, src2, src2stride,

                                             block_w, block_h, mx, my,

                                             lc->mc_buffer);

    }

}
