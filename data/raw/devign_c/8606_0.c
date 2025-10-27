static int mov_write_single_packet(AVFormatContext *s, AVPacket *pkt)

{

        MOVMuxContext *mov = s->priv_data;

        MOVTrack *trk = &mov->tracks[pkt->stream_index];

        AVCodecParameters *par = trk->par;

        int64_t frag_duration = 0;

        int size = pkt->size;



        int ret = check_pkt(s, pkt);

        if (ret < 0)

            return ret;



        if (mov->flags & FF_MOV_FLAG_FRAG_DISCONT) {

            int i;

            for (i = 0; i < s->nb_streams; i++)

                mov->tracks[i].frag_discont = 1;

            mov->flags &= ~FF_MOV_FLAG_FRAG_DISCONT;

        }



        if (!pkt->size) {

            if (trk->start_dts == AV_NOPTS_VALUE && trk->frag_discont) {

                trk->start_dts = pkt->dts;

                if (pkt->pts != AV_NOPTS_VALUE)

                    trk->start_cts = pkt->pts - pkt->dts;

                else

                    trk->start_cts = 0;

            }



            if (trk->par->codec_id == AV_CODEC_ID_MP4ALS ||

                trk->par->codec_id == AV_CODEC_ID_FLAC) {

                int side_size = 0;

                uint8_t *side = av_packet_get_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA, &side_size);

                if (side && side_size > 0 && (side_size != par->extradata_size || memcmp(side, par->extradata, side_size))) {

                    void *newextra = av_mallocz(side_size + AV_INPUT_BUFFER_PADDING_SIZE);

                    if (!newextra)

                        return AVERROR(ENOMEM);

                    av_free(par->extradata);

                    par->extradata = newextra;

                    memcpy(par->extradata, side, side_size);

                    par->extradata_size = side_size;

                    mov->need_rewrite_extradata = 1;

                }

            }



            return 0;             /* Discard 0 sized packets */

        }



        if (trk->entry && pkt->stream_index < s->nb_streams)

            frag_duration = av_rescale_q(pkt->dts - trk->cluster[0].dts,

                                         s->streams[pkt->stream_index]->time_base,

                                         AV_TIME_BASE_Q);

        if ((mov->max_fragment_duration &&

             frag_duration >= mov->max_fragment_duration) ||

             (mov->max_fragment_size && mov->mdat_size + size >= mov->max_fragment_size) ||

             (mov->flags & FF_MOV_FLAG_FRAG_KEYFRAME &&

              par->codec_type == AVMEDIA_TYPE_VIDEO &&

              trk->entry && pkt->flags & AV_PKT_FLAG_KEY)) {

            if (frag_duration >= mov->min_fragment_duration) {

                // Set the duration of this track to line up with the next

                // sample in this track. This avoids relying on AVPacket

                // duration, but only helps for this particular track, not

                // for the other ones that are flushed at the same time.

                trk->track_duration = pkt->dts - trk->start_dts;

                if (pkt->pts != AV_NOPTS_VALUE)

                    trk->end_pts = pkt->pts;

                else

                    trk->end_pts = pkt->dts;

                trk->end_reliable = 1;

                mov_auto_flush_fragment(s, 0);

            }

        }



        return ff_mov_write_packet(s, pkt);

}
