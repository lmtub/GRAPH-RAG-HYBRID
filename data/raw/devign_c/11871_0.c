static void compute_pkt_fields(AVFormatContext *s, AVStream *st,

                               AVCodecParserContext *pc, AVPacket *pkt)

{

    int num, den, presentation_delayed, delay, i;

    int64_t offset;



    if (s->flags & AVFMT_FLAG_NOFILLIN)

        return;



    if((s->flags & AVFMT_FLAG_IGNDTS) && pkt->pts != AV_NOPTS_VALUE)

        pkt->dts= AV_NOPTS_VALUE;



    if (st->codec->codec_id != CODEC_ID_H264 && pc && pc->pict_type == FF_B_TYPE)

        //FIXME Set low_delay = 0 when has_b_frames = 1

        st->codec->has_b_frames = 1;



    /* do we have a video B-frame ? */

    delay= st->codec->has_b_frames;

    presentation_delayed = 0;

    /* XXX: need has_b_frame, but cannot get it if the codec is

        not initialized */

    if (delay &&

        pc && pc->pict_type != FF_B_TYPE)

        presentation_delayed = 1;



    if(pkt->pts != AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE && pkt->dts > pkt->pts && st->pts_wrap_bits<63

       /*&& pkt->dts-(1LL<<st->pts_wrap_bits) < pkt->pts*/){

        pkt->dts -= 1LL<<st->pts_wrap_bits;

    }



    // some mpeg2 in mpeg-ps lack dts (issue171 / input_file.mpg)

    // we take the conservative approach and discard both

    // Note, if this is misbehaving for a H.264 file then possibly presentation_delayed is not set correctly.

    if(delay==1 && pkt->dts == pkt->pts && pkt->dts != AV_NOPTS_VALUE && presentation_delayed){

        av_log(s, AV_LOG_WARNING, "invalid dts/pts combination\n");

        pkt->dts= pkt->pts= AV_NOPTS_VALUE;

    }



    if (pkt->duration == 0) {

        compute_frame_duration(&num, &den, st, pc, pkt);

        if (den && num) {

            pkt->duration = av_rescale_rnd(1, num * (int64_t)st->time_base.den, den * (int64_t)st->time_base.num, AV_ROUND_DOWN);



            if(pkt->duration != 0 && s->packet_buffer)

                update_initial_durations(s, st, pkt);

        }

    }



    /* correct timestamps with byte offset if demuxers only have timestamps

       on packet boundaries */

    if(pc && st->need_parsing == AVSTREAM_PARSE_TIMESTAMPS && pkt->size){

        /* this will estimate bitrate based on this frame's duration and size */

        offset = av_rescale(pc->offset, pkt->duration, pkt->size);

        if(pkt->pts != AV_NOPTS_VALUE)

            pkt->pts += offset;

        if(pkt->dts != AV_NOPTS_VALUE)

            pkt->dts += offset;

    }



    if (pc && pc->dts_sync_point >= 0) {

        // we have synchronization info from the parser

        int64_t den = st->codec->time_base.den * (int64_t) st->time_base.num;

        if (den > 0) {

            int64_t num = st->codec->time_base.num * (int64_t) st->time_base.den;

            if (pkt->dts != AV_NOPTS_VALUE) {

                // got DTS from the stream, update reference timestamp

                st->reference_dts = pkt->dts - pc->dts_ref_dts_delta * num / den;

                pkt->pts = pkt->dts + pc->pts_dts_delta * num / den;

            } else if (st->reference_dts != AV_NOPTS_VALUE) {

                // compute DTS based on reference timestamp

                pkt->dts = st->reference_dts + pc->dts_ref_dts_delta * num / den;

                pkt->pts = pkt->dts + pc->pts_dts_delta * num / den;

            }

            if (pc->dts_sync_point > 0)

                st->reference_dts = pkt->dts; // new reference

        }

    }



    /* This may be redundant, but it should not hurt. */

    if(pkt->dts != AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE && pkt->pts > pkt->dts)

        presentation_delayed = 1;



//    av_log(NULL, AV_LOG_DEBUG, "IN delayed:%d pts:%"PRId64", dts:%"PRId64" cur_dts:%"PRId64" st:%d pc:%p\n", presentation_delayed, pkt->pts, pkt->dts, st->cur_dts, pkt->stream_index, pc);

    /* interpolate PTS and DTS if they are not present */

    //We skip H264 currently because delay and has_b_frames are not reliably set

    if((delay==0 || (delay==1 && pc)) && st->codec->codec_id != CODEC_ID_H264){

        if (presentation_delayed) {

            /* DTS = decompression timestamp */

            /* PTS = presentation timestamp */

            if (pkt->dts == AV_NOPTS_VALUE)

                pkt->dts = st->last_IP_pts;

            update_initial_timestamps(s, pkt->stream_index, pkt->dts, pkt->pts);

            if (pkt->dts == AV_NOPTS_VALUE)

                pkt->dts = st->cur_dts;



            /* this is tricky: the dts must be incremented by the duration

            of the frame we are displaying, i.e. the last I- or P-frame */

            if (st->last_IP_duration == 0)

                st->last_IP_duration = pkt->duration;

            if(pkt->dts != AV_NOPTS_VALUE)

                st->cur_dts = pkt->dts + st->last_IP_duration;

            st->last_IP_duration  = pkt->duration;

            st->last_IP_pts= pkt->pts;

            /* cannot compute PTS if not present (we can compute it only

            by knowing the future */

        } else if(pkt->pts != AV_NOPTS_VALUE || pkt->dts != AV_NOPTS_VALUE || pkt->duration){

            if(pkt->pts != AV_NOPTS_VALUE && pkt->duration){

                int64_t old_diff= FFABS(st->cur_dts - pkt->duration - pkt->pts);

                int64_t new_diff= FFABS(st->cur_dts - pkt->pts);

                if(old_diff < new_diff && old_diff < (pkt->duration>>3)){

                    pkt->pts += pkt->duration;

    //                av_log(NULL, AV_LOG_DEBUG, "id:%d old:%"PRId64" new:%"PRId64" dur:%d cur:%"PRId64" size:%d\n", pkt->stream_index, old_diff, new_diff, pkt->duration, st->cur_dts, pkt->size);

                }

            }



            /* presentation is not delayed : PTS and DTS are the same */

            if(pkt->pts == AV_NOPTS_VALUE)

                pkt->pts = pkt->dts;

            update_initial_timestamps(s, pkt->stream_index, pkt->pts, pkt->pts);

            if(pkt->pts == AV_NOPTS_VALUE)

                pkt->pts = st->cur_dts;

            pkt->dts = pkt->pts;

            if(pkt->pts != AV_NOPTS_VALUE)

                st->cur_dts = pkt->pts + pkt->duration;

        }

    }



    if(pkt->pts != AV_NOPTS_VALUE && delay <= MAX_REORDER_DELAY){

        st->pts_buffer[0]= pkt->pts;

        for(i=0; i<delay && st->pts_buffer[i] > st->pts_buffer[i+1]; i++)

            FFSWAP(int64_t, st->pts_buffer[i], st->pts_buffer[i+1]);

        if(pkt->dts == AV_NOPTS_VALUE)

            pkt->dts= st->pts_buffer[0];

        if(st->codec->codec_id == CODEC_ID_H264){ //we skiped it above so we try here

            update_initial_timestamps(s, pkt->stream_index, pkt->dts, pkt->pts); // this should happen on the first packet

        }

        if(pkt->dts > st->cur_dts)

            st->cur_dts = pkt->dts;

    }



//    av_log(NULL, AV_LOG_ERROR, "OUTdelayed:%d/%d pts:%"PRId64", dts:%"PRId64" cur_dts:%"PRId64"\n", presentation_delayed, delay, pkt->pts, pkt->dts, st->cur_dts);



    /* update flags */

    if(is_intra_only(st->codec))

        pkt->flags |= AV_PKT_FLAG_KEY;

    else if (pc) {

        pkt->flags = 0;

        /* keyframe computation */

        if (pc->key_frame == 1)

            pkt->flags |= AV_PKT_FLAG_KEY;

        else if (pc->key_frame == -1 && pc->pict_type == FF_I_TYPE)

            pkt->flags |= AV_PKT_FLAG_KEY;

    }

    if (pc)

        pkt->convergence_duration = pc->convergence_duration;

}
