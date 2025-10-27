static void lowpass16(WaveformContext *s, AVFrame *in, AVFrame *out,

                      int component, int intensity, int offset, int column)

{

    const int plane = s->desc->comp[component].plane;

    const int mirror = s->mirror;

    const int is_chroma = (component == 1 || component == 2);

    const int shift_w = (is_chroma ? s->desc->log2_chroma_w : 0);

    const int shift_h = (is_chroma ? s->desc->log2_chroma_h : 0);

    const int src_linesize = in->linesize[plane] / 2;

    const int dst_linesize = out->linesize[plane] / 2;

    const int dst_signed_linesize = dst_linesize * (mirror == 1 ? -1 : 1);

    const int limit = s->size - 1;

    const int max = limit - intensity;

    const int src_h = FF_CEIL_RSHIFT(in->height, shift_h);

    const int src_w = FF_CEIL_RSHIFT(in->width, shift_w);

    const uint16_t *src_data = (const uint16_t *)in->data[plane];

    uint16_t *dst_data = (uint16_t *)out->data[plane] + (column ? (offset >> shift_h) * dst_linesize : offset >> shift_w);

    uint16_t * const dst_bottom_line = dst_data + dst_linesize * ((s->size >> shift_h) - 1);

    uint16_t * const dst_line = (mirror ? dst_bottom_line : dst_data);

    const uint16_t *p;

    int y;



    if (!column && mirror)

        dst_data += s->size >> shift_w;



    for (y = 0; y < src_h; y++) {

        const uint16_t *src_data_end = src_data + src_w;

        uint16_t *dst = dst_line;



        for (p = src_data; p < src_data_end; p++) {

            uint16_t *target;

            int v = FFMIN(*p, limit);



            if (column) {

                target = dst++ + dst_signed_linesize * (v >> shift_h);

            } else {

                if (mirror)

                    target = dst_data - (v >> shift_w) - 1;

                else

                    target = dst_data + (v >> shift_w);

            }

            update16(target, max, intensity, limit);

        }

        src_data += src_linesize;

        dst_data += dst_linesize;

    }



    envelope16(s, out, plane, plane);

}
