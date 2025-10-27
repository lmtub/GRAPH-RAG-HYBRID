void copy_picture_field(TInterlaceContext *tinterlace,

                        uint8_t *dst[4], int dst_linesize[4],

                        const uint8_t *src[4], int src_linesize[4],

                        enum AVPixelFormat format, int w, int src_h,

                        int src_field, int interleave, int dst_field,

                        int flags)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);

    int hsub = desc->log2_chroma_w;

    int plane, vsub = desc->log2_chroma_h;

    int k = src_field == FIELD_UPPER_AND_LOWER ? 1 : 2;

    int h;



    for (plane = 0; plane < desc->nb_components; plane++) {

        int lines = plane == 1 || plane == 2 ? AV_CEIL_RSHIFT(src_h, vsub) : src_h;

        int cols  = plane == 1 || plane == 2 ? AV_CEIL_RSHIFT(    w, hsub) : w;

        uint8_t *dstp = dst[plane];

        const uint8_t *srcp = src[plane];

        int srcp_linesize = src_linesize[plane] * k;

        int dstp_linesize = dst_linesize[plane] * (interleave ? 2 : 1);



        lines = (lines + (src_field == FIELD_UPPER)) / k;

        if (src_field == FIELD_LOWER)

            srcp += src_linesize[plane];

        if (interleave && dst_field == FIELD_LOWER)

            dstp += dst_linesize[plane];

        // Low-pass filtering is required when creating an interlaced destination from

        // a progressive source which contains high-frequency vertical detail.

        // Filtering will reduce interlace 'twitter' and Moire patterning.

        if (flags & TINTERLACE_FLAG_VLPF || flags & TINTERLACE_FLAG_CVLPF) {

            int x = 0;

            if (flags & TINTERLACE_FLAG_CVLPF)

                x = 1;

            for (h = lines; h > 0; h--) {

                ptrdiff_t pref = src_linesize[plane];

                ptrdiff_t mref = -pref;

                if (h >= (lines - x))  mref = 0; // there is no line above

                else if (h <= (1 + x)) pref = 0; // there is no line below



                tinterlace->lowpass_line(dstp, cols, srcp, mref, pref);

                dstp += dstp_linesize;

                srcp += srcp_linesize;

            }

        } else {

            av_image_copy_plane(dstp, dstp_linesize, srcp, srcp_linesize, cols, lines);

        }

    }

}
