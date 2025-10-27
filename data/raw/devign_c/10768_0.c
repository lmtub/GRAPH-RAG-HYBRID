static void achroma(WaveformContext *s, AVFrame *in, AVFrame *out,

                    int component, int intensity, int offset, int column)

{

    const int plane = s->desc->comp[component].plane;

    const int mirror = s->mirror;

    const int c1_linesize = in->linesize[(plane + 1) % s->ncomp];

    const int c2_linesize = in->linesize[(plane + 2) % s->ncomp];

    const int d1_linesize = out->linesize[(plane + 1) % s->ncomp];

    const int d2_linesize = out->linesize[(plane + 2) % s->ncomp];

    const int max = 255 - intensity;

    const int src_h = in->height;

    const int src_w = in->width;

    int x, y;



    if (column) {

        const int d1_signed_linesize = d1_linesize * (mirror == 1 ? -1 : 1);

        const int d2_signed_linesize = d2_linesize * (mirror == 1 ? -1 : 1);



        for (x = 0; x < src_w; x++) {

            const uint8_t *c1_data = in->data[(plane + 1) % s->ncomp];

            const uint8_t *c2_data = in->data[(plane + 2) % s->ncomp];

            uint8_t *d1_data = out->data[(plane + 1) % s->ncomp] + offset * d1_linesize;

            uint8_t *d2_data = out->data[(plane + 2) % s->ncomp] + offset * d2_linesize;

            uint8_t * const d1_bottom_line = d1_data + d1_linesize * (s->size - 1);

            uint8_t * const d1 = (mirror ? d1_bottom_line : d1_data);

            uint8_t * const d2_bottom_line = d2_data + d2_linesize * (s->size - 1);

            uint8_t * const d2 = (mirror ? d2_bottom_line : d2_data);



            for (y = 0; y < src_h; y++) {

                const int c1 = c1_data[x] - 128;

                const int c2 = c2_data[x] - 128;

                uint8_t *target;

                int p;



                for (p = 128 + c1; p < 128; p++) {

                    target = d1 + x + d1_signed_linesize * p;

                    update(target, max, 1);

                }



                for (p = 128 + c1 - 1; p > 128; p--) {

                    target = d1 + x + d1_signed_linesize * p;

                    update(target, max, 1);

                }



                for (p = 128 + c2; p < 128; p++) {

                    target = d2 + x + d2_signed_linesize * p;

                    update(target, max, 1);

                }



                for (p = 128 + c2 - 1; p > 128; p--) {

                    target = d2 + x + d2_signed_linesize * p;

                    update(target, max, 1);

                }



                c1_data += c1_linesize;

                c2_data += c2_linesize;

                d1_data += d1_linesize;

                d2_data += d2_linesize;

            }

        }

    } else {

        const uint8_t *c1_data = in->data[(plane + 1) % s->ncomp];

        const uint8_t *c2_data = in->data[(plane + 2) % s->ncomp];

        uint8_t *d0_data = out->data[plane] + offset;

        uint8_t *d1_data = out->data[(plane + 1) % s->ncomp] + offset;

        uint8_t *d2_data = out->data[(plane + 2) % s->ncomp] + offset;



        if (mirror) {

            d0_data += s->size - 1;

            d1_data += s->size - 1;

            d2_data += s->size - 1;

        }



        for (y = 0; y < src_h; y++) {

            for (x = 0; x < src_w; x++) {

                const int c1 = c1_data[x] - 128;

                const int c2 = c2_data[x] - 128;

                uint8_t *target;

                int p;



                for (p = 128 + c1; p < 128; p++) {

                    if (mirror)

                        target = d1_data - p;

                    else

                        target = d1_data + p;



                    update(target, max, 1);

                }



                for (p = 128 + 1; p < 128 + c1; p++) {

                    if (mirror)

                        target = d1_data - p;

                    else

                        target = d1_data + p;



                    update(target, max, 1);

                }



                for (p = 128 + c2; p < 128; p++) {

                    if (mirror)

                        target = d2_data - p;

                    else

                        target = d2_data + p;



                    update(target, max, 1);

                }



                for (p = 128 + 1; p < 128 + c2; p++) {

                    if (mirror)

                        target = d2_data - p;

                    else

                        target = d2_data + p;



                    update(target, max, 1);

                }

            }



            c1_data += c1_linesize;

            c2_data += c2_linesize;

            d1_data += d1_linesize;

            d2_data += d2_linesize;

        }

    }



    envelope(s, out, plane, (plane + 1) % s->ncomp);

    envelope(s, out, plane, (plane + 2) % s->ncomp);

}
