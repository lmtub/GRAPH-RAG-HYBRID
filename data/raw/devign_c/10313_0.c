static void get_slice_data(ProresContext *ctx, const uint16_t *src,

                           int linesize, int x, int y, int w, int h,

                           DCTELEM *blocks, uint16_t *emu_buf,

                           int mbs_per_slice, int blocks_per_mb, int is_chroma)

{

    const uint16_t *esrc;

    const int mb_width = 4 * blocks_per_mb;

    int elinesize;

    int i, j, k;



    for (i = 0; i < mbs_per_slice; i++, src += mb_width) {

        if (x >= w) {

            memset(blocks, 0, 64 * (mbs_per_slice - i) * blocks_per_mb

                              * sizeof(*blocks));

            return;

        }

        if (x + mb_width <= w && y + 16 <= h) {

            esrc      = src;

            elinesize = linesize;

        } else {

            int bw, bh, pix;



            esrc      = emu_buf;

            elinesize = 16 * sizeof(*emu_buf);



            bw = FFMIN(w - x, mb_width);

            bh = FFMIN(h - y, 16);



            for (j = 0; j < bh; j++) {

                memcpy(emu_buf + j * 16,

                       (const uint8_t*)src + j * linesize,

                       bw * sizeof(*src));

                pix = emu_buf[j * 16 + bw - 1];

                for (k = bw; k < mb_width; k++)

                    emu_buf[j * 16 + k] = pix;

            }

            for (; j < 16; j++)

                memcpy(emu_buf + j * 16,

                       emu_buf + (bh - 1) * 16,

                       mb_width * sizeof(*emu_buf));

        }

        if (!is_chroma) {

            ctx->dsp.fdct(esrc, elinesize, blocks);

            blocks += 64;

            if (blocks_per_mb > 2) {

                ctx->dsp.fdct(src + 8, linesize, blocks);

                blocks += 64;

            }

            ctx->dsp.fdct(src + linesize * 4, linesize, blocks);

            blocks += 64;

            if (blocks_per_mb > 2) {

                ctx->dsp.fdct(src + linesize * 4 + 8, linesize, blocks);

                blocks += 64;

            }

        } else {

            ctx->dsp.fdct(esrc, elinesize, blocks);

            blocks += 64;

            ctx->dsp.fdct(src + linesize * 4, linesize, blocks);

            blocks += 64;

            if (blocks_per_mb > 2) {

                ctx->dsp.fdct(src + 8, linesize, blocks);

                blocks += 64;

                ctx->dsp.fdct(src + linesize * 4 + 8, linesize, blocks);

                blocks += 64;

            }

        }



        x += mb_width;

    }

}
