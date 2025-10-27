static int deband_16_coupling_c(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)

{

    DebandContext *s = ctx->priv;

    ThreadData *td = arg;

    AVFrame *in = td->in;

    AVFrame *out = td->out;

    const int start = (s->planeheight[0] *  jobnr   ) / nb_jobs;

    const int end   = (s->planeheight[0] * (jobnr+1)) / nb_jobs;

    int x, y, p, z;



    for (y = start; y < end; y++) {

        const int pos = y * s->planewidth[0];



        for (x = 0; x < s->planewidth[p]; x++) {

            const int x_pos = s->x_pos[pos + x];

            const int y_pos = s->y_pos[pos + x];

            int avg[4], cmp[4] = { 0 }, src[4];



            for (p = 0; p < s->nb_components; p++) {

                const uint16_t *src_ptr = (const uint16_t *)in->data[p];

                const int src_linesize = in->linesize[p] / 2;

                const int thr = s->thr[p];

                const int w = s->planewidth[p] - 1;

                const int h = s->planeheight[p] - 1;

                const int ref0 = src_ptr[av_clip(y +  y_pos, 0, h) * src_linesize + av_clip(x +  x_pos, 0, w)];

                const int ref1 = src_ptr[av_clip(y + -y_pos, 0, h) * src_linesize + av_clip(x +  x_pos, 0, w)];

                const int ref2 = src_ptr[av_clip(y + -y_pos, 0, h) * src_linesize + av_clip(x + -x_pos, 0, w)];

                const int ref3 = src_ptr[av_clip(y +  y_pos, 0, h) * src_linesize + av_clip(x + -x_pos, 0, w)];

                const int src0 = src_ptr[y * src_linesize + x];



                src[p] = src0;

                avg[p] = get_avg(ref0, ref1, ref2, ref3);



                if (s->blur) {

                    cmp[p] = FFABS(src0 - avg[p]) < thr;

                } else {

                    cmp[p] = (FFABS(src0 - ref0) < thr) &&

                             (FFABS(src0 - ref1) < thr) &&

                             (FFABS(src0 - ref2) < thr) &&

                             (FFABS(src0 - ref3) < thr);

                }

            }



            for (z = 0; z < s->nb_components; z++)

                if (!cmp[z])

                    break;

            if (z == s->nb_components) {

                for (p = 0; p < s->nb_components; p++) {

                    const int dst_linesize = out->linesize[p] / 2;

                    uint16_t *dst = (uint16_t *)out->data[p] + y * dst_linesize + x;



                    dst[0] = avg[p];

                }

            } else {

                for (p = 0; p < s->nb_components; p++) {

                    const int dst_linesize = out->linesize[p] / 2;

                    uint16_t *dst = (uint16_t *)out->data[p] + y * dst_linesize + x;



                    dst[0] = src[p];

                }

            }

        }

    }



    return 0;

}
