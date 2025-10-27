static void envelope_instant16(WaveformContext *s, AVFrame *out, int plane, int component)

{

    const int dst_linesize = out->linesize[component] / 2;

    const int bg = s->bg_color[component] * (s->size / 256);

    const int limit = s->size - 1;

    const int is_chroma = (component == 1 || component == 2);

    const int shift_w = (is_chroma ? s->desc->log2_chroma_w : 0);

    const int shift_h = (is_chroma ? s->desc->log2_chroma_h : 0);

    const int dst_h = FF_CEIL_RSHIFT(out->height, shift_h);

    const int dst_w = FF_CEIL_RSHIFT(out->width, shift_w);

    const int start = s->estart[plane];

    const int end = s->eend[plane];

    uint16_t *dst;

    int x, y;



    if (s->mode) {

        for (x = 0; x < dst_w; x++) {

            for (y = start; y < end; y++) {

                dst = (uint16_t *)out->data[component] + y * dst_linesize + x;

                if (dst[0] != bg) {

                    dst[0] = limit;

                    break;

                }

            }

            for (y = end - 1; y >= start; y--) {

                dst = (uint16_t *)out->data[component] + y * dst_linesize + x;

                if (dst[0] != bg) {

                    dst[0] = limit;

                    break;

                }

            }

        }

    } else {

        for (y = 0; y < dst_h; y++) {

            dst = (uint16_t *)out->data[component] + y * dst_linesize;

            for (x = start; x < end; x++) {

                if (dst[x] != bg) {

                    dst[x] = limit;

                    break;

                }

            }

            for (x = end - 1; x >= start; x--) {

                if (dst[x] != bg) {

                    dst[x] = limit;

                    break;

                }

            }

        }

    }

}
