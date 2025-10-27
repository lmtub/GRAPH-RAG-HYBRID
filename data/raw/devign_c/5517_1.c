static int init_pass2(MpegEncContext *s)

{

    RateControlContext *rcc = &s->rc_context;

    AVCodecContext *a       = s->avctx;

    int i, toobig;

    double fps             = get_fps(s->avctx);

    double complexity[5]   = { 0 }; // approximate bits at quant=1

    uint64_t const_bits[5] = { 0 }; // quantizer independent bits

    uint64_t all_const_bits;

    uint64_t all_available_bits = (uint64_t)(s->bit_rate *

                                             (double)rcc->num_entries / fps);

    double rate_factor          = 0;

    double step;

    const int filter_size = (int)(a->qblur * 4) | 1;

    double expected_bits;

    double *qscale, *blurred_qscale, qscale_sum;



    /* find complexity & const_bits & decide the pict_types */

    for (i = 0; i < rcc->num_entries; i++) {

        RateControlEntry *rce = &rcc->entry[i];



        rce->new_pict_type                = rce->pict_type;

        rcc->i_cplx_sum[rce->pict_type]  += rce->i_tex_bits * rce->qscale;

        rcc->p_cplx_sum[rce->pict_type]  += rce->p_tex_bits * rce->qscale;

        rcc->mv_bits_sum[rce->pict_type] += rce->mv_bits;

        rcc->frame_count[rce->pict_type]++;



        complexity[rce->new_pict_type] += (rce->i_tex_bits + rce->p_tex_bits) *

                                          (double)rce->qscale;

        const_bits[rce->new_pict_type] += rce->mv_bits + rce->misc_bits;

    }



    all_const_bits = const_bits[AV_PICTURE_TYPE_I] +

                     const_bits[AV_PICTURE_TYPE_P] +

                     const_bits[AV_PICTURE_TYPE_B];



    if (all_available_bits < all_const_bits) {

        av_log(s->avctx, AV_LOG_ERROR, "requested bitrate is too low\n");

        return -1;

    }



    qscale         = av_malloc(sizeof(double) * rcc->num_entries);

    blurred_qscale = av_malloc(sizeof(double) * rcc->num_entries);

    toobig = 0;



    for (step = 256 * 256; step > 0.0000001; step *= 0.5) {

        expected_bits = 0;

        rate_factor  += step;



        rcc->buffer_index = s->avctx->rc_buffer_size / 2;



        /* find qscale */

        for (i = 0; i < rcc->num_entries; i++) {

            RateControlEntry *rce = &rcc->entry[i];



            qscale[i] = get_qscale(s, &rcc->entry[i], rate_factor, i);

            rcc->last_qscale_for[rce->pict_type] = qscale[i];

        }

        assert(filter_size % 2 == 1);



        /* fixed I/B QP relative to P mode */

        for (i = FFMAX(0, rcc->num_entries - 300); i < rcc->num_entries; i++) {

            RateControlEntry *rce = &rcc->entry[i];



            qscale[i] = get_diff_limited_q(s, rce, qscale[i]);

        }



        for (i = rcc->num_entries - 1; i >= 0; i--) {

            RateControlEntry *rce = &rcc->entry[i];



            qscale[i] = get_diff_limited_q(s, rce, qscale[i]);

        }



        /* smooth curve */

        for (i = 0; i < rcc->num_entries; i++) {

            RateControlEntry *rce = &rcc->entry[i];

            const int pict_type   = rce->new_pict_type;

            int j;

            double q = 0.0, sum = 0.0;



            for (j = 0; j < filter_size; j++) {

                int index    = i + j - filter_size / 2;

                double d     = index - i;

                double coeff = a->qblur == 0 ? 1.0 : exp(-d * d / (a->qblur * a->qblur));



                if (index < 0 || index >= rcc->num_entries)

                    continue;

                if (pict_type != rcc->entry[index].new_pict_type)

                    continue;

                q   += qscale[index] * coeff;

                sum += coeff;

            }

            blurred_qscale[i] = q / sum;

        }



        /* find expected bits */

        for (i = 0; i < rcc->num_entries; i++) {

            RateControlEntry *rce = &rcc->entry[i];

            double bits;



            rce->new_qscale = modify_qscale(s, rce, blurred_qscale[i], i);



            bits  = qp2bits(rce, rce->new_qscale) + rce->mv_bits + rce->misc_bits;

            bits += 8 * ff_vbv_update(s, bits);



            rce->expected_bits = expected_bits;

            expected_bits     += bits;

        }



        av_dlog(s->avctx,

                "expected_bits: %f all_available_bits: %d rate_factor: %f\n",

                expected_bits, (int)all_available_bits, rate_factor);

        if (expected_bits > all_available_bits) {

            rate_factor -= step;

            ++toobig;

        }

    }

    av_free(qscale);

    av_free(blurred_qscale);



    /* check bitrate calculations and print info */

    qscale_sum = 0.0;

    for (i = 0; i < rcc->num_entries; i++) {

        av_dlog(s, "[lavc rc] entry[%d].new_qscale = %.3f  qp = %.3f\n",

                i,

                rcc->entry[i].new_qscale,

                rcc->entry[i].new_qscale / FF_QP2LAMBDA);

        qscale_sum += av_clip(rcc->entry[i].new_qscale / FF_QP2LAMBDA,

                              s->avctx->qmin, s->avctx->qmax);

    }

    assert(toobig <= 40);

    av_log(s->avctx, AV_LOG_DEBUG,

           "[lavc rc] requested bitrate: %d bps  expected bitrate: %d bps\n",

           s->bit_rate,

           (int)(expected_bits / ((double)all_available_bits / s->bit_rate)));

    av_log(s->avctx, AV_LOG_DEBUG,

           "[lavc rc] estimated target average qp: %.3f\n",

           (float)qscale_sum / rcc->num_entries);

    if (toobig == 0) {

        av_log(s->avctx, AV_LOG_INFO,

               "[lavc rc] Using all of requested bitrate is not "

               "necessary for this video with these parameters.\n");

    } else if (toobig == 40) {

        av_log(s->avctx, AV_LOG_ERROR,

               "[lavc rc] Error: bitrate too low for this video "

               "with these parameters.\n");

        return -1;

    } else if (fabs(expected_bits / all_available_bits - 1.0) > 0.01) {

        av_log(s->avctx, AV_LOG_ERROR,

               "[lavc rc] Error: 2pass curve failed to converge\n");

        return -1;

    }



    return 0;

}
