static void filter(AVFilterContext *ctx, AVFilterBufferRef *dstpic,

                   int parity, int tff)

{

    YADIFContext *yadif = ctx->priv;

    int y, i;



    for (i = 0; i < yadif->csp->nb_components; i++) {

        int w = dstpic->video->w;

        int h = dstpic->video->h;

        int refs = yadif->cur->linesize[i];

        int df = (yadif->csp->comp[i].depth_minus1 + 8) / 8;



        if (i == 1 || i == 2) {

        /* Why is this not part of the per-plane description thing? */

            w >>= yadif->csp->log2_chroma_w;

            h >>= yadif->csp->log2_chroma_h;

        }



        for (y = 0; y < h; y++) {

            if ((y ^ parity) & 1) {

                uint8_t *prev = &yadif->prev->data[i][y*refs];

                uint8_t *cur  = &yadif->cur ->data[i][y*refs];

                uint8_t *next = &yadif->next->data[i][y*refs];

                uint8_t *dst  = &dstpic->data[i][y*dstpic->linesize[i]];

                int     mode  = y==1 || y+2==h ? 2 : yadif->mode;

                yadif->filter_line(dst, prev, cur, next, w, y+1<h ? refs : -refs, y ? -refs : refs, parity ^ tff, mode);

            } else {

                memcpy(&dstpic->data[i][y*dstpic->linesize[i]],

                       &yadif->cur->data[i][y*refs], w*df);

            }

        }

    }



    emms_c();

}
