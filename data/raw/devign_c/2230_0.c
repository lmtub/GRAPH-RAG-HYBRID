static void compute_pts_dts(AVStream *st, int64_t *ppts, int64_t *pdts, 

                            int64_t timestamp)

{

    int frame_delay;

    int64_t pts, dts;



    if (st->codec.codec_type == CODEC_TYPE_VIDEO && 

        st->codec.max_b_frames != 0) {

        frame_delay = (st->codec.frame_rate_base * 90000LL) / 

            st->codec.frame_rate;

        if (timestamp == 0) {

            /* specific case for first frame : DTS just before */

            pts = timestamp;

            dts = timestamp - frame_delay;

        } else {

            timestamp -= frame_delay;

            if (st->codec.coded_frame->pict_type == FF_B_TYPE) {

                /* B frames has identical pts/dts */

                pts = timestamp;

                dts = timestamp;

            } else {

                /* a reference frame has a pts equal to the dts of the

                   _next_ one */

                dts = timestamp;

                pts = timestamp + (st->codec.max_b_frames + 1) * frame_delay;

            }

        }

#if 1

        av_log(&st->codec, AV_LOG_DEBUG, "pts=%0.3f dts=%0.3f pict_type=%c\n", 

               pts / 90000.0, dts / 90000.0, 

               av_get_pict_type_char(st->codec.coded_frame->pict_type));

#endif

    } else {

        pts = timestamp;

        dts = timestamp;

    }



    *ppts = pts & ((1LL << 33) - 1);

    *pdts = dts & ((1LL << 33) - 1);

}
