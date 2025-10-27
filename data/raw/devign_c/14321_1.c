static int mov_read_colr(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    char color_parameter_type[5] = { 0 };

    int color_primaries, color_trc, color_matrix;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams - 1];



    avio_read(pb, color_parameter_type, 4);

    if (strncmp(color_parameter_type, "nclx", 4) &&

        strncmp(color_parameter_type, "nclc", 4)) {

        av_log(c->fc, AV_LOG_WARNING, "unsupported color_parameter_type %s\n",

               color_parameter_type);

        return 0;

    }



    color_primaries = avio_rb16(pb);

    color_trc = avio_rb16(pb);

    color_matrix = avio_rb16(pb);



    av_log(c->fc, AV_LOG_TRACE,

           "%s: pri %"PRIu16" trc %"PRIu16" matrix %"PRIu16"",

           color_parameter_type, color_primaries, color_trc, color_matrix);



    if (!strncmp(color_parameter_type, "nclx", 4)) {

        uint8_t color_range = avio_r8(pb) >> 7;

        av_log(c->fc, AV_LOG_TRACE, " full %"PRIu8"", color_range);

        if (color_range)

            st->codec->color_range = AVCOL_RANGE_JPEG;

        else

            st->codec->color_range = AVCOL_RANGE_MPEG;

        /* 14496-12 references JPEG XR specs (rather than the more complete

         * 23001-8) so some adjusting is required */

        if (color_primaries >= AVCOL_PRI_FILM)

            color_primaries = AVCOL_PRI_UNSPECIFIED;

        if ((color_trc >= AVCOL_TRC_LINEAR &&

             color_trc <= AVCOL_TRC_LOG_SQRT) ||

            color_trc >= AVCOL_TRC_BT2020_10)

            color_trc = AVCOL_TRC_UNSPECIFIED;

        if (color_matrix >= AVCOL_SPC_BT2020_NCL)

            color_matrix = AVCOL_SPC_UNSPECIFIED;

        st->codec->color_primaries = color_primaries;

        st->codec->color_trc = color_trc;

        st->codec->colorspace = color_matrix;

    } else if (!strncmp(color_parameter_type, "nclc", 4)) {

        /* color primaries, Table 4-4 */

        switch (color_primaries) {

        case 1: st->codec->color_primaries = AVCOL_PRI_BT709; break;

        case 5: st->codec->color_primaries = AVCOL_PRI_SMPTE170M; break;

        case 6: st->codec->color_primaries = AVCOL_PRI_SMPTE240M; break;

        }

        /* color transfer, Table 4-5 */

        switch (color_trc) {

        case 1: st->codec->color_trc = AVCOL_TRC_BT709; break;

        case 7: st->codec->color_trc = AVCOL_TRC_SMPTE240M; break;

        }

        /* color matrix, Table 4-6 */

        switch (color_matrix) {

        case 1: st->codec->colorspace = AVCOL_SPC_BT709; break;

        case 6: st->codec->colorspace = AVCOL_SPC_BT470BG; break;

        case 7: st->codec->colorspace = AVCOL_SPC_SMPTE240M; break;

        }

    }

    av_log(c->fc, AV_LOG_TRACE, "\n");



    return 0;

}
