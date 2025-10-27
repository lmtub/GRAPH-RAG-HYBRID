static void noise(uint8_t *dst, const uint8_t *src,

                  int dst_linesize, int src_linesize,

                  int width, int start, int end, NoiseContext *n, int comp)

{

    FilterParams *p = &n->param[comp];

    int8_t *noise = p->noise;

    const int flags = p->flags;

    AVLFG *lfg = &p->lfg;

    int shift, y;



    if (!noise) {

        if (dst != src)

            av_image_copy_plane(dst, dst_linesize, src, src_linesize, width, end - start);

        return;

    }



    for (y = start; y < end; y++) {

        if (flags & NOISE_TEMPORAL)

            shift = av_lfg_get(lfg) & (MAX_SHIFT - 1);

        else

            shift = n->rand_shift[y];



        if (flags & NOISE_AVERAGED) {

            n->line_noise_avg(dst, src, width, p->prev_shift[y]);

            p->prev_shift[y][shift & 3] = noise + shift;

        } else {

            n->line_noise(dst, src, noise, width, shift);

        }

        dst += dst_linesize;

        src += src_linesize;

    }

}
