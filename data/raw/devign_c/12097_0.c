static int do_packet_auto_bsf(AVFormatContext *s, AVPacket *pkt) {

    AVStream *st = s->streams[pkt->stream_index];

    int i, ret;



    if (!(s->flags & AVFMT_FLAG_AUTO_BSF))

        return 1;



    if (s->oformat->check_bitstream) {

        if (!st->internal->bitstream_checked) {

            if ((ret = s->oformat->check_bitstream(s, pkt)) < 0)

                return ret;

            else if (ret == 1)

                st->internal->bitstream_checked = 1;

        }

    }



#if FF_API_LAVF_MERGE_SD

FF_DISABLE_DEPRECATION_WARNINGS

    if (st->internal->nb_bsfcs) {

        ret = av_packet_split_side_data(pkt);

        if (ret < 0)

            av_log(s, AV_LOG_WARNING, "Failed to split side data before bitstream filter\n");

    }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    for (i = 0; i < st->internal->nb_bsfcs; i++) {

        AVBSFContext *ctx = st->internal->bsfcs[i];

        if (i > 0) {

            AVBSFContext* prev_ctx = st->internal->bsfcs[i - 1];

            if (prev_ctx->par_out->extradata_size != ctx->par_in->extradata_size) {

                if ((ret = avcodec_parameters_copy(ctx->par_in, prev_ctx->par_out)) < 0)

                    return ret;

            }

        }

        // TODO: when any bitstream filter requires flushing at EOF, we'll need to

        // flush each stream's BSF chain on write_trailer.

        if ((ret = av_bsf_send_packet(ctx, pkt)) < 0) {

            av_log(ctx, AV_LOG_ERROR,

                    "Failed to send packet to filter %s for stream %d\n",

                    ctx->filter->name, pkt->stream_index);

            return ret;

        }

        // TODO: when any automatically-added bitstream filter is generating multiple

        // output packets for a single input one, we'll need to call this in a loop

        // and write each output packet.

        if ((ret = av_bsf_receive_packet(ctx, pkt)) < 0) {

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)

                return 0;

            av_log(ctx, AV_LOG_ERROR,

                    "Failed to send packet to filter %s for stream %d\n",

                    ctx->filter->name, pkt->stream_index);

            return ret;

        }

        if (i == st->internal->nb_bsfcs - 1) {

            if (ctx->par_out->extradata_size != st->codecpar->extradata_size) {

                if ((ret = avcodec_parameters_copy(st->codecpar, ctx->par_out)) < 0)

                    return ret;

            }

        }

    }

    return 1;

}
