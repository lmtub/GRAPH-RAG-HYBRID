static int ffm_seek(AVFormatContext *s, int stream_index, int64_t wanted_pts, int flags)

{

    FFMContext *ffm = s->priv_data;

    int64_t pos_min, pos_max, pos;

    int64_t pts_min, pts_max, pts;

    double pos1;



    av_dlog(s, "wanted_pts=%0.6f\n", wanted_pts / 1000000.0);

    /* find the position using linear interpolation (better than

       dichotomy in typical cases) */

    if (ffm->write_index && ffm->write_index < ffm->file_size) {

        if (get_dts(s, FFM_PACKET_SIZE) < wanted_pts) {

            pos_min = FFM_PACKET_SIZE;

            pos_max = ffm->write_index - FFM_PACKET_SIZE;

        } else {

            pos_min = ffm->write_index;

            pos_max = ffm->file_size - FFM_PACKET_SIZE;

        }

    } else {

        pos_min = FFM_PACKET_SIZE;

        pos_max = ffm->file_size - FFM_PACKET_SIZE;

    }

    while (pos_min <= pos_max) {

        pts_min = get_dts(s, pos_min);

        pts_max = get_dts(s, pos_max);

        if (pts_min > wanted_pts || pts_max < wanted_pts) {

            pos = pts_min > wanted_pts ? pos_min : pos_max;

            goto found;

        }

        /* linear interpolation */

        pos1 = (double)(pos_max - pos_min) * (double)(wanted_pts - pts_min) /

            (double)(pts_max - pts_min);

        pos = (((int64_t)pos1) / FFM_PACKET_SIZE) * FFM_PACKET_SIZE;

        if (pos <= pos_min)

            pos = pos_min;

        else if (pos >= pos_max)

            pos = pos_max;

        pts = get_dts(s, pos);

        /* check if we are lucky */

        if (pts == wanted_pts) {

            goto found;

        } else if (pts > wanted_pts) {

            pos_max = pos - FFM_PACKET_SIZE;

        } else {

            pos_min = pos + FFM_PACKET_SIZE;

        }

    }

    pos = (flags & AVSEEK_FLAG_BACKWARD) ? pos_min : pos_max;



 found:

    if (ffm_seek1(s, pos) < 0)

        return -1;



    /* reset read state */

    ffm->read_state = READ_HEADER;

    ffm->packet_ptr = ffm->packet;

    ffm->packet_end = ffm->packet;

    ffm->first_packet = 1;



    return 0;

}
