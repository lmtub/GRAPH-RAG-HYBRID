static int seg_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    SegmentContext *seg = s->priv_data;

    AVStream *st = s->streams[pkt->stream_index];

    int64_t end_pts = INT64_MAX, offset;

    int start_frame = INT_MAX;

    int ret;

    struct tm ti;

    int64_t usecs;

    int64_t wrapped_val;



    if (!seg->avf)

        return AVERROR(EINVAL);



calc_times:

    if (seg->times) {

        end_pts = seg->segment_count < seg->nb_times ?

            seg->times[seg->segment_count] : INT64_MAX;

    } else if (seg->frames) {

        start_frame = seg->segment_count < seg->nb_frames ?

            seg->frames[seg->segment_count] : INT_MAX;

    } else {

        if (seg->use_clocktime) {

            int64_t avgt = av_gettime();

            time_t sec = avgt / 1000000;

            localtime_r(&sec, &ti);

            usecs = (int64_t)(ti.tm_hour * 3600 + ti.tm_min * 60 + ti.tm_sec) * 1000000 + (avgt % 1000000);

            wrapped_val = (usecs + seg->clocktime_offset) % seg->time;

            if (seg->last_cut != usecs && wrapped_val < seg->last_val && wrapped_val < seg->clocktime_wrap_duration) {

                seg->cut_pending = 1;

                seg->last_cut = usecs;

            }

            seg->last_val = wrapped_val;

        } else {

            end_pts = seg->time * (seg->segment_count + 1);

        }

    }



    ff_dlog(s, "packet stream:%d pts:%s pts_time:%s duration_time:%s is_key:%d frame:%d\n",

            pkt->stream_index, av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base),

            av_ts2timestr(pkt->duration, &st->time_base),

            pkt->flags & AV_PKT_FLAG_KEY,

            pkt->stream_index == seg->reference_stream_index ? seg->frame_count : -1);



    if (pkt->stream_index == seg->reference_stream_index &&

        (pkt->flags & AV_PKT_FLAG_KEY || seg->break_non_keyframes) &&

        (seg->segment_frame_count > 0 || seg->write_empty) &&

        (seg->cut_pending || seg->frame_count >= start_frame ||

         (pkt->pts != AV_NOPTS_VALUE &&

          av_compare_ts(pkt->pts, st->time_base,

                        end_pts - seg->time_delta, AV_TIME_BASE_Q) >= 0))) {

        /* sanitize end time in case last packet didn't have a defined duration */

        if (seg->cur_entry.last_duration == 0)

            seg->cur_entry.end_time = (double)pkt->pts * av_q2d(st->time_base);



        if ((ret = segment_end(s, seg->individual_header_trailer, 0)) < 0)

            goto fail;



        if ((ret = segment_start(s, seg->individual_header_trailer)) < 0)

            goto fail;



        seg->cut_pending = 0;

        seg->cur_entry.index = seg->segment_idx + seg->segment_idx_wrap * seg->segment_idx_wrap_nb;

        seg->cur_entry.start_time = (double)pkt->pts * av_q2d(st->time_base);

        seg->cur_entry.start_pts = av_rescale_q(pkt->pts, st->time_base, AV_TIME_BASE_Q);

        seg->cur_entry.end_time = seg->cur_entry.start_time;



        if (seg->times || (!seg->frames && !seg->use_clocktime) && seg->write_empty)

            goto calc_times;

    }



    if (pkt->stream_index == seg->reference_stream_index) {

        if (pkt->pts != AV_NOPTS_VALUE)

            seg->cur_entry.end_time =

                FFMAX(seg->cur_entry.end_time, (double)(pkt->pts + pkt->duration) * av_q2d(st->time_base));

        seg->cur_entry.last_duration = pkt->duration;

    }



    if (seg->segment_frame_count == 0) {

        av_log(s, AV_LOG_VERBOSE, "segment:'%s' starts with packet stream:%d pts:%s pts_time:%s frame:%d\n",

               seg->avf->filename, pkt->stream_index,

               av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base), seg->frame_count);

    }



    av_log(s, AV_LOG_DEBUG, "stream:%d start_pts_time:%s pts:%s pts_time:%s dts:%s dts_time:%s",

           pkt->stream_index,

           av_ts2timestr(seg->cur_entry.start_pts, &AV_TIME_BASE_Q),

           av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base),

           av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &st->time_base));



    /* compute new timestamps */

    offset = av_rescale_q(seg->initial_offset - (seg->reset_timestamps ? seg->cur_entry.start_pts : 0),

                          AV_TIME_BASE_Q, st->time_base);

    if (pkt->pts != AV_NOPTS_VALUE)

        pkt->pts += offset;

    if (pkt->dts != AV_NOPTS_VALUE)

        pkt->dts += offset;



    av_log(s, AV_LOG_DEBUG, " -> pts:%s pts_time:%s dts:%s dts_time:%s\n",

           av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &st->time_base),

           av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &st->time_base));



    ret = ff_write_chained(seg->avf, pkt->stream_index, pkt, s, seg->initial_offset || seg->reset_timestamps);



fail:

    if (pkt->stream_index == seg->reference_stream_index) {

        seg->frame_count++;

        seg->segment_frame_count++;

    }



    return ret;

}
