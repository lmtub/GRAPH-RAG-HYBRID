int av_interleaved_write_frame(AVFormatContext *s, AVPacket *pkt)

{

    int ret, flush = 0;



    ret = check_packet(s, pkt);

    if (ret < 0)

        goto fail;



    if (pkt) {

        AVStream *st = s->streams[pkt->stream_index];



        //FIXME/XXX/HACK drop zero sized packets

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && pkt->size == 0) {

            ret = 0;

            goto fail;

        }



        av_dlog(s, "av_interleaved_write_frame size:%d dts:%" PRId64 " pts:%" PRId64 "\n",

                pkt->size, pkt->dts, pkt->pts);

        if ((ret = compute_pkt_fields2(s, st, pkt)) < 0 && !(s->oformat->flags & AVFMT_NOTIMESTAMPS))

            goto fail;



        if (pkt->dts == AV_NOPTS_VALUE && !(s->oformat->flags & AVFMT_NOTIMESTAMPS)) {

            ret = AVERROR(EINVAL);

            goto fail;

        }

    } else {

        av_dlog(s, "av_interleaved_write_frame FLUSH\n");

        flush = 1;

    }



    for (;; ) {

        AVPacket opkt;

        int ret = interleave_packet(s, &opkt, pkt, flush);

        if (pkt) {

            memset(pkt, 0, sizeof(*pkt));

            av_init_packet(pkt);

            pkt = NULL;

        }

        if (ret <= 0) //FIXME cleanup needed for ret<0 ?

            return ret;



        ret = write_packet(s, &opkt);

        if (ret >= 0)

            s->streams[opkt.stream_index]->nb_frames++;



        av_free_packet(&opkt);



        if (ret < 0)

            return ret;

    }

fail:

    av_packet_unref(pkt);

    return ret;

}
