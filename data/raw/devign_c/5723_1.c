static int hls_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    HLSContext *hls = s->priv_data;

    AVFormatContext *oc = NULL;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t end_pts = 0;

    int is_ref_pkt = 1;

    int ret = 0, can_split = 1, i, j;

    int stream_index = 0;

    int range_length = 0;

    uint8_t *buffer = NULL;

    VariantStream *vs = NULL;



    for (i = 0; i < hls->nb_varstreams; i++) {

        vs = &hls->var_streams[i];

        for (j = 0; j < vs->nb_streams; j++) {

            if (vs->streams[j] == st) {

                if( st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE ) {

                    oc = vs->vtt_avf;

                    stream_index = 0;

                } else {

                    oc = vs->avf;

                    stream_index = j;

                }

                break;

            }

        }



        if (oc)

            break;

    }



    if (!oc) {

        av_log(s, AV_LOG_ERROR, "Unable to find mapping variant stream\n");

        return AVERROR(ENOMEM);

    }



    end_pts = hls->recording_time * vs->number;



    if (vs->sequence - vs->nb_entries > hls->start_sequence && hls->init_time > 0) {

        /* reset end_pts, hls->recording_time at end of the init hls list */

        int init_list_dur = hls->init_time * vs->nb_entries * AV_TIME_BASE;

        int after_init_list_dur = (vs->sequence - vs->nb_entries ) * hls->time * AV_TIME_BASE;

        hls->recording_time = hls->time * AV_TIME_BASE;

        end_pts = init_list_dur + after_init_list_dur ;

    }



    if (vs->start_pts == AV_NOPTS_VALUE) {

        vs->start_pts = pkt->pts;

        vs->end_pts   = pkt->pts;

    }



    if (vs->has_video) {

        can_split = st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&

                    ((pkt->flags & AV_PKT_FLAG_KEY) || (hls->flags & HLS_SPLIT_BY_TIME));

        is_ref_pkt = st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;

    }

    if (pkt->pts == AV_NOPTS_VALUE)

        is_ref_pkt = can_split = 0;



    if (is_ref_pkt) {

        if (vs->new_start) {

            vs->new_start = 0;

            vs->duration = (double)(pkt->pts - vs->end_pts)

                                       * st->time_base.num / st->time_base.den;

            vs->dpp = (double)(pkt->duration) * st->time_base.num / st->time_base.den;

        } else {

            if (pkt->duration) {

                vs->duration += (double)(pkt->duration) * st->time_base.num / st->time_base.den;

            } else {

                av_log(s, AV_LOG_WARNING, "pkt->duration = 0, maybe the hls segment duration will not precise\n");

                vs->duration = (double)(pkt->pts - vs->end_pts) * st->time_base.num / st->time_base.den;

            }

        }



    }



    if (vs->packets_written && can_split && av_compare_ts(pkt->pts - vs->start_pts, st->time_base,

                                   end_pts, AV_TIME_BASE_Q) >= 0) {

        int64_t new_start_pos;

        char *old_filename = av_strdup(vs->avf->filename);

        int byterange_mode = (hls->flags & HLS_SINGLE_FILE) || (hls->max_seg_size > 0);



        if (!old_filename) {

            return AVERROR(ENOMEM);

        }



        av_write_frame(vs->avf, NULL); /* Flush any buffered data */



        new_start_pos = avio_tell(vs->avf->pb);

        vs->size = new_start_pos - vs->start_pos;



        if (!byterange_mode) {

            if (hls->segment_type == SEGMENT_TYPE_FMP4 && !vs->init_range_length) {

                avio_flush(oc->pb);

                range_length = avio_close_dyn_buf(oc->pb, &buffer);

                avio_write(vs->out, buffer, range_length);

                vs->init_range_length = range_length;

                avio_open_dyn_buf(&oc->pb);

                vs->packets_written = 0;

                ff_format_io_close(s, &vs->out);

            } else {

                ff_format_io_close(s, &oc->pb);

            }

            if (vs->vtt_avf) {

                ff_format_io_close(s, &vs->vtt_avf->pb);

            }

        }

        if ((hls->flags & HLS_TEMP_FILE) && oc->filename[0]) {

            if (!(hls->flags & HLS_SINGLE_FILE) || (hls->max_seg_size <= 0))

                if ((vs->avf->oformat->priv_class && vs->avf->priv_data) && hls->segment_type != SEGMENT_TYPE_FMP4)

                    av_opt_set(vs->avf->priv_data, "mpegts_flags", "resend_headers", 0);

            hls_rename_temp_file(s, oc);

        }



        if (vs->fmp4_init_mode) {

            vs->number--;

        }



        if (!vs->fmp4_init_mode || byterange_mode)

            ret = hls_append_segment(s, hls, vs, vs->duration, vs->start_pos, vs->size);

        vs->start_pos = new_start_pos;

        if (ret < 0) {

            av_free(old_filename);

            return ret;

        }



        vs->end_pts = pkt->pts;

        vs->duration = 0;



        vs->fmp4_init_mode = 0;

        if (hls->flags & HLS_SINGLE_FILE) {

            vs->number++;

        } else if (hls->max_seg_size > 0) {

            if (vs->start_pos >= hls->max_seg_size) {

                vs->sequence++;

                sls_flag_file_rename(hls, vs, old_filename);

                ret = hls_start(s, vs);

                vs->start_pos = 0;

                /* When split segment by byte, the duration is short than hls_time,

                 * so it is not enough one segment duration as hls_time, */

                vs->number--;

            }

            vs->number++;

        } else {

            sls_flag_file_rename(hls, vs, old_filename);

            ret = hls_start(s, vs);

        }

        av_free(old_filename);



        if (ret < 0) {

            return ret;

        }



        if (!vs->fmp4_init_mode || byterange_mode)

            if ((ret = hls_window(s, 0, vs)) < 0) {

                return ret;

            }

    }



    vs->packets_written++;

    ret = ff_write_chained(oc, stream_index, pkt, s, 0);



    return ret;

}
