static int hls_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    HLSContext *hls = s->priv_data;

    AVFormatContext *oc = NULL;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t end_pts = hls->recording_time * hls->number;

    int is_ref_pkt = 1;

    int ret, can_split = 1;

    int stream_index = 0;



    if (hls->sequence - hls->nb_entries > hls->start_sequence && hls->init_time > 0) {

        /* reset end_pts, hls->recording_time at end of the init hls list */

        int init_list_dur = hls->init_time * hls->nb_entries * AV_TIME_BASE;

        int after_init_list_dur = (hls->sequence - hls->nb_entries ) * hls->time * AV_TIME_BASE;

        hls->recording_time = hls->time * AV_TIME_BASE;

        end_pts = init_list_dur + after_init_list_dur ;

    }



    if( st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE ) {

        oc = hls->vtt_avf;

        stream_index = 0;

    } else {

        oc = hls->avf;

        stream_index = pkt->stream_index;

    }

    if (hls->start_pts == AV_NOPTS_VALUE) {

        hls->start_pts = pkt->pts;

        hls->end_pts   = pkt->pts;

    }



    if (hls->has_video) {

        can_split = st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&

                    ((pkt->flags & AV_PKT_FLAG_KEY) || (hls->flags & HLS_SPLIT_BY_TIME));

        is_ref_pkt = st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;

    }

    if (pkt->pts == AV_NOPTS_VALUE)

        is_ref_pkt = can_split = 0;



    if (is_ref_pkt) {

        if (hls->new_start) {

            hls->new_start = 0;

            hls->duration = (double)(pkt->pts - hls->end_pts)

                                       * st->time_base.num / st->time_base.den;

            hls->dpp = (double)(pkt->duration) * st->time_base.num / st->time_base.den;

        } else {

            hls->duration += (double)(pkt->duration) * st->time_base.num / st->time_base.den;

        }



    }

    if (can_split && av_compare_ts(pkt->pts - hls->start_pts, st->time_base,

                                   end_pts, AV_TIME_BASE_Q) >= 0) {

        int64_t new_start_pos;

        char *old_filename = av_strdup(hls->avf->filename);



        if (!old_filename) {

            return AVERROR(ENOMEM);

        }



        av_write_frame(oc, NULL); /* Flush any buffered data */



        new_start_pos = avio_tell(hls->avf->pb);

        hls->size = new_start_pos - hls->start_pos;



        ff_format_io_close(s, &oc->pb);

        if (hls->vtt_avf) {

            ff_format_io_close(s, &hls->vtt_avf->pb);

        }

        if ((hls->flags & HLS_TEMP_FILE) && oc->filename[0]) {

            if (!(hls->flags & HLS_SINGLE_FILE) || (hls->max_seg_size <= 0))

                if (hls->avf->oformat->priv_class && hls->avf->priv_data)

                    av_opt_set(hls->avf->priv_data, "mpegts_flags", "resend_headers", 0);

            hls_rename_temp_file(s, oc);

        }



        ret = hls_append_segment(s, hls, hls->duration, hls->start_pos, hls->size);

        hls->start_pos = new_start_pos;

        if (ret < 0) {

            av_free(old_filename);

            return ret;

        }



        hls->end_pts = pkt->pts;

        hls->duration = 0;



        if (hls->flags & HLS_SINGLE_FILE) {

            hls->number++;

        } else if (hls->max_seg_size > 0) {

            if (hls->start_pos >= hls->max_seg_size) {

                hls->sequence++;

                sls_flag_file_rename(hls, old_filename);

                ret = hls_start(s);

                hls->start_pos = 0;

                /* When split segment by byte, the duration is short than hls_time,

                 * so it is not enough one segment duration as hls_time, */

                hls->number--;

            }

            hls->number++;

        } else {

            sls_flag_file_rename(hls, old_filename);

            ret = hls_start(s);

        }



        if (ret < 0) {

            av_free(old_filename);

            return ret;

        }



        if ((ret = hls_window(s, 0)) < 0) {

            av_free(old_filename);

            return ret;

        }

    }



    ret = ff_write_chained(oc, stream_index, pkt, s, 0);



    return ret;

}
