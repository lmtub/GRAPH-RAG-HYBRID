static int asf_read_seek(AVFormatContext *s, int stream_index, int64_t pts)

{

    ASFContext *asf = s->priv_data;

    AVStream *st;

    AVPacket pkt1, *pkt;

    int block_align;

    int64_t pos;

    int64_t pos_min, pos_max, pts_min, pts_max, cur_pts;



    pkt = &pkt1;



    // Validate pts

    if (pts < 0)

	pts = 0;



    if (stream_index == -1)

        stream_index= av_find_default_stream_index(s);

    

    st = s->streams[stream_index];



    // ASF files have fixed block sizes, store this to determine offset

    block_align = asf->packet_size;



    if (block_align <= 0)

        return -1;



    pos_min = 0;

    pts_min = asf_read_pts(s, &pos_min, stream_index);

    if (pts_min == AV_NOPTS_VALUE) return -1;

   

    pos_max = asf_align(s, url_filesize(url_fileno(&s->pb)) - 1 - s->data_offset); //FIXME wrong

    pts_max = pts_min + s->duration;



    while (pos_min <= pos_max) {



        if (pts <= pts_min) {

            pos = pos_min;

            goto found;

        } else if (pts >= pts_max) {

            pos = pos_max;

            goto found;

        } else {

            // interpolate position (better than dichotomy)

            pos = (int64_t)((double)(pos_max - pos_min) *

                            (double)(pts - pts_min) /

                            (double)(pts_max - pts_min)) + pos_min;

            pos= asf_align(s, pos);

        }



        // read the next timestamp 

    	cur_pts = asf_read_pts(s, &pos, stream_index);    

	

        /* check if we are lucky */

        if (pts == cur_pts) {

            goto found;

        } else if (cur_pts == AV_NOPTS_VALUE) {

	    return -1;

        } else if (pts < cur_pts) {

            pos_max = pos;

	    pts_max = asf_read_pts(s, &pos_max, stream_index); //FIXME wrong, must do backward search, or change this somehow

            if (pts >= pts_max) {

                pos = pos_max;

                goto found;

            }

        } else {

            pos_min = pos + asf->packet_size;

	    pts_min = asf_read_pts(s, &pos_min, stream_index);

            if (pts <= pts_min) {

                goto found;

            }

        }

    }

    pos = pos_min;

found:

    url_fseek(&s->pb, pos + s->data_offset, SEEK_SET);

    asf_reset_header(s);

    return 0;

}
