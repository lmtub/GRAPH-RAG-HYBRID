static int mpegps_read_seek(AVFormatContext *s, 

                            int stream_index, int64_t timestamp)

{

    int64_t pos_min, pos_max, pos;

    int64_t dts_min, dts_max, dts;



    timestamp = (timestamp * 90000) / AV_TIME_BASE;



#ifdef DEBUG_SEEK

    printf("read_seek: %d %0.3f\n", stream_index, timestamp / 90000.0);

#endif



    /* XXX: find stream_index by looking at the first PES packet found */

    if (stream_index < 0) {

        stream_index = find_stream_index(s);

        if (stream_index < 0)

            return -1;

    }

    pos_min = 0;

    dts_min = mpegps_read_dts(s, stream_index, &pos_min, 1);

    if (dts_min == AV_NOPTS_VALUE) {

        /* we can reach this case only if no PTS are present in

           the whole stream */

        return -1;

    }

    pos_max = url_filesize(url_fileno(&s->pb)) - 1;

    dts_max = mpegps_read_dts(s, stream_index, &pos_max, 0);

    

    while (pos_min <= pos_max) {

#ifdef DEBUG_SEEK

        printf("pos_min=0x%llx pos_max=0x%llx dts_min=%0.3f dts_max=%0.3f\n", 

               pos_min, pos_max,

               dts_min / 90000.0, dts_max / 90000.0);

#endif

        if (timestamp <= dts_min) {

            pos = pos_min;

            goto found;

        } else if (timestamp >= dts_max) {

            pos = pos_max;

            goto found;

        } else {

            /* interpolate position (better than dichotomy) */

            pos = (int64_t)((double)(pos_max - pos_min) * 

                            (double)(timestamp - dts_min) /

                            (double)(dts_max - dts_min)) + pos_min;

        }

#ifdef DEBUG_SEEK

        printf("pos=0x%llx\n", pos);

#endif

        /* read the next timestamp */

        dts = mpegps_read_dts(s, stream_index, &pos, 1);

        /* check if we are lucky */

        if (dts == AV_NOPTS_VALUE) {

            /* should never happen */

            pos = pos_min;

            goto found;

        } else if (timestamp == dts) {

            goto found;

        } else if (timestamp < dts) {

            pos_max = pos;

            dts_max = mpegps_read_dts(s, stream_index, &pos_max, 0);

            if (dts_max == AV_NOPTS_VALUE) {

                /* should never happen */

                break;

            } else if (timestamp >= dts_max) {

                pos = pos_max;

                goto found;

            }

        } else {

            pos_min = pos + 1;

            dts_min = mpegps_read_dts(s, stream_index, &pos_min, 1);

            if (dts_min == AV_NOPTS_VALUE) {

                /* should never happen */

                goto found;

            } else if (timestamp <= dts_min) {

                goto found;

            }

        }

    }

    pos = pos_min;

 found:

#ifdef DEBUG_SEEK

    pos_min = pos;

    dts_min = mpegps_read_dts(s, stream_index, &pos_min, 1);

    pos_min++;

    dts_max = mpegps_read_dts(s, stream_index, &pos_min, 1);

    printf("pos=0x%llx %0.3f<=%0.3f<=%0.3f\n", 

           pos, dts_min / 90000.0, timestamp / 90000.0, dts_max / 90000.0);

#endif

    /* do the seek */

    url_fseek(&s->pb, pos, SEEK_SET);

    return 0;

}
