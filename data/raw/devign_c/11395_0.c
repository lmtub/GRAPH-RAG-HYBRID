static int vp7_decode_frame_header(VP8Context *s, const uint8_t *buf, int buf_size)

{

    VP56RangeCoder *c = &s->c;

    int part1_size, hscale, vscale, i, j, ret;

    int width  = s->avctx->width;

    int height = s->avctx->height;



    s->profile = (buf[0] >> 1) & 7;

    if (s->profile > 1) {

        avpriv_request_sample(s->avctx, "Unknown profile %d", s->profile);

        return AVERROR_INVALIDDATA;

    }



    s->keyframe  = !(buf[0] & 1);

    s->invisible = 0;

    part1_size   = AV_RL24(buf) >> 4;



    buf      += 4 - s->profile;

    buf_size -= 4 - s->profile;



    memcpy(s->put_pixels_tab, s->vp8dsp.put_vp8_epel_pixels_tab, sizeof(s->put_pixels_tab));



    ff_vp56_init_range_decoder(c, buf, part1_size);

    buf      += part1_size;

    buf_size -= part1_size;



    /* A. Dimension information (keyframes only) */

    if (s->keyframe) {

        width  = vp8_rac_get_uint(c, 12);

        height = vp8_rac_get_uint(c, 12);

        hscale = vp8_rac_get_uint(c, 2);

        vscale = vp8_rac_get_uint(c, 2);

        if (hscale || vscale)

            avpriv_request_sample(s->avctx, "Upscaling");



        s->update_golden = s->update_altref = VP56_FRAME_CURRENT;

        vp78_reset_probability_tables(s);

        memcpy(s->prob->pred16x16, vp8_pred16x16_prob_inter,

               sizeof(s->prob->pred16x16));

        memcpy(s->prob->pred8x8c, vp8_pred8x8c_prob_inter,

               sizeof(s->prob->pred8x8c));

        for (i = 0; i < 2; i++)

            memcpy(s->prob->mvc[i], vp7_mv_default_prob[i],

                   sizeof(vp7_mv_default_prob[i]));

        memset(&s->segmentation, 0, sizeof(s->segmentation));

        memset(&s->lf_delta, 0, sizeof(s->lf_delta));

        memcpy(s->prob[0].scan, zigzag_scan, sizeof(s->prob[0].scan));

    }



    if (s->keyframe || s->profile > 0)

        memset(s->inter_dc_pred, 0 , sizeof(s->inter_dc_pred));



    /* B. Decoding information for all four macroblock-level features */

    for (i = 0; i < 4; i++) {

        s->feature_enabled[i] = vp8_rac_get(c);

        if (s->feature_enabled[i]) {

             s->feature_present_prob[i] = vp8_rac_get_uint(c, 8);



             for (j = 0; j < 3; j++)

                 s->feature_index_prob[i][j] =

                     vp8_rac_get(c) ? vp8_rac_get_uint(c, 8) : 255;



             if (vp7_feature_value_size[i])

                 for (j = 0; j < 4; j++)

                     s->feature_value[i][j] =

                         vp8_rac_get(c) ? vp8_rac_get_uint(c, vp7_feature_value_size[s->profile][i]) : 0;

        }

    }



    s->segmentation.enabled    = 0;

    s->segmentation.update_map = 0;

    s->lf_delta.enabled        = 0;



    s->num_coeff_partitions = 1;

    ff_vp56_init_range_decoder(&s->coeff_partition[0], buf, buf_size);



    if (!s->macroblocks_base || /* first frame */

        width != s->avctx->width || height != s->avctx->height ||

        (width + 15) / 16 != s->mb_width || (height + 15) / 16 != s->mb_height) {

        if ((ret = vp7_update_dimensions(s, width, height)) < 0)

            return ret;

    }



    /* C. Dequantization indices */

    vp7_get_quants(s);



    /* D. Golden frame update flag (a Flag) for interframes only */

    if (!s->keyframe) {

        s->update_golden = vp8_rac_get(c) ? VP56_FRAME_CURRENT : VP56_FRAME_NONE;

        s->sign_bias[VP56_FRAME_GOLDEN] = 0;

    }



    s->update_last          = 1;

    s->update_probabilities = 1;

    s->fade_present         = 1;



    if (s->profile > 0) {

        s->update_probabilities = vp8_rac_get(c);

        if (!s->update_probabilities)

            s->prob[1] = s->prob[0];



        if (!s->keyframe)

            s->fade_present = vp8_rac_get(c);

    }



    /* E. Fading information for previous frame */

    if (s->fade_present && vp8_rac_get(c)) {

        if ((ret = vp7_fade_frame(s ,c)) < 0)

            return ret;

    }



    /* F. Loop filter type */

    if (!s->profile)

        s->filter.simple = vp8_rac_get(c);



    /* G. DCT coefficient ordering specification */

    if (vp8_rac_get(c))

        for (i = 1; i < 16; i++)

            s->prob[0].scan[i] = zigzag_scan[vp8_rac_get_uint(c, 4)];



    /* H. Loop filter levels  */

    if (s->profile > 0)

        s->filter.simple = vp8_rac_get(c);

    s->filter.level     = vp8_rac_get_uint(c, 6);

    s->filter.sharpness = vp8_rac_get_uint(c, 3);



    /* I. DCT coefficient probability update; 13.3 Token Probability Updates */

    vp78_update_probability_tables(s);



    s->mbskip_enabled = 0;



    /* J. The remaining frame header data occurs ONLY FOR INTERFRAMES */

    if (!s->keyframe) {

        s->prob->intra  = vp8_rac_get_uint(c, 8);

        s->prob->last   = vp8_rac_get_uint(c, 8);

        vp78_update_pred16x16_pred8x8_mvc_probabilities(s, VP7_MVC_SIZE);

    }



    return 0;

}
