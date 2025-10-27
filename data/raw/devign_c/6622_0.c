static void copy_picture_field(InterlaceContext *s,

                               AVFrame *src_frame, AVFrame *dst_frame,

                               AVFilterLink *inlink, enum FieldType field_type,

                               int lowpass)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    int hsub = desc->log2_chroma_w;

    int vsub = desc->log2_chroma_h;

    int plane, j;



    for (plane = 0; plane < desc->nb_components; plane++) {

        int cols  = (plane == 1 || plane == 2) ? -(-inlink->w) >> hsub : inlink->w;

        int lines = (plane == 1 || plane == 2) ? -(-inlink->h) >> vsub : inlink->h;

        uint8_t *dstp = dst_frame->data[plane];

        const uint8_t *srcp = src_frame->data[plane];



        av_assert0(cols >= 0 || lines >= 0);



        lines = (lines + (field_type == FIELD_UPPER)) / 2;

        if (field_type == FIELD_LOWER)

            srcp += src_frame->linesize[plane];

        if (field_type == FIELD_LOWER)

            dstp += dst_frame->linesize[plane];

        if (lowpass) {

            int srcp_linesize = src_frame->linesize[plane] * 2;

            int dstp_linesize = dst_frame->linesize[plane] * 2;

            for (j = lines; j > 0; j--) {

                const uint8_t *srcp_above = srcp - src_frame->linesize[plane];

                const uint8_t *srcp_below = srcp + src_frame->linesize[plane];

                if (j == lines)

                    srcp_above = srcp; // there is no line above

                if (j == 1)

                    srcp_below = srcp; // there is no line below

                s->lowpass_line(dstp, cols, srcp, srcp_above, srcp_below);

                dstp += dstp_linesize;

                srcp += srcp_linesize;

            }

        } else {

            av_image_copy_plane(dstp, dst_frame->linesize[plane] * 2,

                                srcp, src_frame->linesize[plane] * 2,

                                cols, lines);

        }

    }

}
