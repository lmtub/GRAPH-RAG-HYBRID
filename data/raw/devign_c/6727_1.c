static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)

{

    YADIFContext *s = ctx->priv;

    ThreadData *td  = arg;

    int refs = s->cur->linesize[td->plane];

    int df = (s->csp->comp[td->plane].depth_minus1 + 8) / 8;

    int pix_3 = 3 * df;

    int slice_h = td->h / nb_jobs;

    int slice_start = jobnr * slice_h;

    int slice_end   = (jobnr == nb_jobs - 1) ? td->h : (jobnr + 1) * slice_h;

    int y;



    /* filtering reads 3 pixels to the left/right; to avoid invalid reads,

     * we need to call the c variant which avoids this for border pixels

     */

    for (y = slice_start; y < slice_end; y++) {

        if ((y ^ td->parity) & 1) {

            uint8_t *prev = &s->prev->data[td->plane][y * refs];

            uint8_t *cur  = &s->cur ->data[td->plane][y * refs];

            uint8_t *next = &s->next->data[td->plane][y * refs];

            uint8_t *dst  = &td->frame->data[td->plane][y * td->frame->linesize[td->plane]];

            int     mode  = y == 1 || y + 2 == td->h ? 2 : s->mode;

            s->filter_line(dst + pix_3, prev + pix_3, cur + pix_3,

                           next + pix_3, td->w - 6,

                           y + 1 < td->h ? refs : -refs,

                           y ? -refs : refs,

                           td->parity ^ td->tff, mode);

            s->filter_edges(dst, prev, cur, next, td->w,

                            y + 1 < td->h ? refs : -refs,

                            y ? -refs : refs,

                            td->parity ^ td->tff, mode);

        } else {

            memcpy(&td->frame->data[td->plane][y * td->frame->linesize[td->plane]],

                   &s->cur->data[td->plane][y * refs], td->w * df);

        }

    }

    return 0;

}
