static int parse_source_parameters(AVCodecContext *avctx, GetBitContext *gb,

                                   dirac_source_params *source)

{

    AVRational frame_rate = {0,0};

    unsigned luma_depth = 8, luma_offset = 16;

    int idx;

    int chroma_x_shift, chroma_y_shift;



    /* [DIRAC_STD] 10.3.2 Frame size. frame_size(video_params) */

    /* [DIRAC_STD] custom_dimensions_flag */

    if (get_bits1(gb)) {

        source->width  = svq3_get_ue_golomb(gb); /* [DIRAC_STD] FRAME_WIDTH  */

        source->height = svq3_get_ue_golomb(gb); /* [DIRAC_STD] FRAME_HEIGHT */

    }



    /* [DIRAC_STD] 10.3.3 Chroma Sampling Format.

     *  chroma_sampling_format(video_params) */

    /* [DIRAC_STD] custom_chroma_format_flag */

    if (get_bits1(gb))

        /* [DIRAC_STD] CHROMA_FORMAT_INDEX */

        source->chroma_format = svq3_get_ue_golomb(gb);

    if (source->chroma_format > 2U) {

        av_log(avctx, AV_LOG_ERROR, "Unknown chroma format %d\n",

               source->chroma_format);

        return AVERROR_INVALIDDATA;

    }



    /* [DIRAC_STD] 10.3.4 Scan Format. scan_format(video_params) */

    /* [DIRAC_STD] custom_scan_format_flag */

    if (get_bits1(gb))

        /* [DIRAC_STD] SOURCE_SAMPLING */

        source->interlaced = svq3_get_ue_golomb(gb);

    if (source->interlaced > 1U)

        return AVERROR_INVALIDDATA;



    /* [DIRAC_STD] 10.3.5 Frame Rate. frame_rate(video_params) */

    if (get_bits1(gb)) { /* [DIRAC_STD] custom_frame_rate_flag */

        source->frame_rate_index = svq3_get_ue_golomb(gb);



        if (source->frame_rate_index > 10U)

            return AVERROR_INVALIDDATA;



        if (!source->frame_rate_index) {

            /* [DIRAC_STD] FRAME_RATE_NUMER */

            frame_rate.num = svq3_get_ue_golomb(gb);

            /* [DIRAC_STD] FRAME_RATE_DENOM */

            frame_rate.den = svq3_get_ue_golomb(gb);

        }

    }

    /* [DIRAC_STD] preset_frame_rate(video_params, index) */

    if (source->frame_rate_index > 0) {

        if (source->frame_rate_index <= 8)

            frame_rate = ff_mpeg12_frame_rate_tab[source->frame_rate_index];

        else

             /* [DIRAC_STD] Table 10.3 values 9-10 */

            frame_rate = dirac_frame_rate[source->frame_rate_index-9];

    }

    av_reduce(&avctx->time_base.num, &avctx->time_base.den,

              frame_rate.den, frame_rate.num, 1<<30);



    /* [DIRAC_STD] 10.3.6 Pixel Aspect Ratio.

     * pixel_aspect_ratio(video_params) */

    if (get_bits1(gb)) { /* [DIRAC_STD] custom_pixel_aspect_ratio_flag */

         /* [DIRAC_STD] index */

        source->aspect_ratio_index = svq3_get_ue_golomb(gb);



        if (source->aspect_ratio_index > 6U)

            return AVERROR_INVALIDDATA;



        if (!source->aspect_ratio_index) {

            avctx->sample_aspect_ratio.num = svq3_get_ue_golomb(gb);

            avctx->sample_aspect_ratio.den = svq3_get_ue_golomb(gb);

        }

    }

    /* [DIRAC_STD] Take value from Table 10.4 Available preset pixel

     *  aspect ratio values */

    if (source->aspect_ratio_index > 0)

        avctx->sample_aspect_ratio =

            dirac_preset_aspect_ratios[source->aspect_ratio_index-1];



    /* [DIRAC_STD] 10.3.7 Clean area. clean_area(video_params) */

    if (get_bits1(gb)) { /* [DIRAC_STD] custom_clean_area_flag */

        /* [DIRAC_STD] CLEAN_WIDTH */

        source->clean_width        = svq3_get_ue_golomb(gb);

         /* [DIRAC_STD] CLEAN_HEIGHT */

        source->clean_height       = svq3_get_ue_golomb(gb);

         /* [DIRAC_STD] CLEAN_LEFT_OFFSET */

        source->clean_left_offset  = svq3_get_ue_golomb(gb);

        /* [DIRAC_STD] CLEAN_RIGHT_OFFSET */

        source->clean_right_offset = svq3_get_ue_golomb(gb);

    }



    /* [DIRAC_STD] 10.3.8 Signal range. signal_range(video_params)

     * WARNING: Some adaptation seems to be done using the

     * AVCOL_RANGE_MPEG/JPEG values */

    if (get_bits1(gb)) { /* [DIRAC_STD] custom_signal_range_flag */

        /* [DIRAC_STD] index */

        source->pixel_range_index = svq3_get_ue_golomb(gb);



        if (source->pixel_range_index > 4U)

            return AVERROR_INVALIDDATA;



        /* This assumes either fullrange or MPEG levels only */

        if (!source->pixel_range_index) {

            luma_offset = svq3_get_ue_golomb(gb);

            luma_depth  = av_log2(svq3_get_ue_golomb(gb))+1;

            svq3_get_ue_golomb(gb); /* chroma offset    */

            svq3_get_ue_golomb(gb); /* chroma excursion */

            avctx->color_range = luma_offset ? AVCOL_RANGE_MPEG : AVCOL_RANGE_JPEG;

        }

    }

    /* [DIRAC_STD] Table 10.5

     * Available signal range presets <--> pixel_range_presets */

    if (source->pixel_range_index > 0) {

        idx                = source->pixel_range_index-1;

        luma_depth         = pixel_range_presets[idx].bitdepth;

        avctx->color_range = pixel_range_presets[idx].color_range;

    }



    if (luma_depth > 8)

        av_log(avctx, AV_LOG_WARNING, "Bitdepth greater than 8\n");



    avctx->pix_fmt = dirac_pix_fmt[!luma_offset][source->chroma_format];

    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &chroma_x_shift, &chroma_y_shift);

    if (!(source->width % (1<<chroma_x_shift)) || !(source->height % (1<<chroma_y_shift))) {

        av_log(avctx, AV_LOG_ERROR, "Dimensions must be a integer multiply of the chroma subsampling\n");

        return AVERROR_INVALIDDATA;

    }





    /* [DIRAC_STD] 10.3.9 Colour specification. colour_spec(video_params) */

    if (get_bits1(gb)) { /* [DIRAC_STD] custom_colour_spec_flag */

         /* [DIRAC_STD] index */

        idx = source->color_spec_index = svq3_get_ue_golomb(gb);



        if (source->color_spec_index > 4U)

            return AVERROR_INVALIDDATA;



        avctx->color_primaries = dirac_color_presets[idx].color_primaries;

        avctx->colorspace      = dirac_color_presets[idx].colorspace;

        avctx->color_trc       = dirac_color_presets[idx].color_trc;



        if (!source->color_spec_index) {

            /* [DIRAC_STD] 10.3.9.1 Colour primaries */

            if (get_bits1(gb)) {

                idx = svq3_get_ue_golomb(gb);

                if (idx < 3U)

                    avctx->color_primaries = dirac_primaries[idx];

            }

            /* [DIRAC_STD] 10.3.9.2 Colour matrix */

            if (get_bits1(gb)) {

                idx = svq3_get_ue_golomb(gb);

                if (!idx)

                    avctx->colorspace = AVCOL_SPC_BT709;

                else if (idx == 1)

                    avctx->colorspace = AVCOL_SPC_BT470BG;

            }

            /* [DIRAC_STD] 10.3.9.3 Transfer function */

            if (get_bits1(gb) && !svq3_get_ue_golomb(gb))

                avctx->color_trc = AVCOL_TRC_BT709;

        }

    } else {

        idx = source->color_spec_index;

        avctx->color_primaries = dirac_color_presets[idx].color_primaries;

        avctx->colorspace      = dirac_color_presets[idx].colorspace;

        avctx->color_trc       = dirac_color_presets[idx].color_trc;

    }



    return 0;

}
