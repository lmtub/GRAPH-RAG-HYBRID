static int libvorbis_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                  const AVFrame *frame, int *got_packet_ptr)

{

    LibvorbisContext *s = avctx->priv_data;

    ogg_packet op;

    int ret, duration;



    /* send samples to libvorbis */

    if (frame) {

        const int samples = frame->nb_samples;

        float **buffer;

        int c, channels = s->vi.channels;



        buffer = vorbis_analysis_buffer(&s->vd, samples);

        for (c = 0; c < channels; c++) {

            int co = (channels > 8) ? c :

                     ff_vorbis_encoding_channel_layout_offsets[channels - 1][c];

            memcpy(buffer[c], frame->extended_data[co],

                   samples * sizeof(*buffer[c]));

        }

        if ((ret = vorbis_analysis_wrote(&s->vd, samples)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "error in vorbis_analysis_wrote()\n");

            return vorbis_error_to_averror(ret);

        }

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0)

            return ret;

    } else {

        if (!s->eof)

            if ((ret = vorbis_analysis_wrote(&s->vd, 0)) < 0) {

                av_log(avctx, AV_LOG_ERROR, "error in vorbis_analysis_wrote()\n");

                return vorbis_error_to_averror(ret);

            }

        s->eof = 1;

    }



    /* retrieve available packets from libvorbis */

    while ((ret = vorbis_analysis_blockout(&s->vd, &s->vb)) == 1) {

        if ((ret = vorbis_analysis(&s->vb, NULL)) < 0)

            break;

        if ((ret = vorbis_bitrate_addblock(&s->vb)) < 0)

            break;



        /* add any available packets to the output packet buffer */

        while ((ret = vorbis_bitrate_flushpacket(&s->vd, &op)) == 1) {

            if (av_fifo_space(s->pkt_fifo) < sizeof(ogg_packet) + op.bytes) {

                av_log(avctx, AV_LOG_ERROR, "packet buffer is too small");

                return AVERROR_BUG;

            }

            av_fifo_generic_write(s->pkt_fifo, &op, sizeof(ogg_packet), NULL);

            av_fifo_generic_write(s->pkt_fifo, op.packet, op.bytes, NULL);

        }

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "error getting available packets\n");

            break;

        }

    }

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "error getting available packets\n");

        return vorbis_error_to_averror(ret);

    }



    /* check for available packets */

    if (av_fifo_size(s->pkt_fifo) < sizeof(ogg_packet))

        return 0;



    av_fifo_generic_read(s->pkt_fifo, &op, sizeof(ogg_packet), NULL);



    if ((ret = ff_alloc_packet(avpkt, op.bytes))) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

        return ret;

    }

    av_fifo_generic_read(s->pkt_fifo, avpkt->data, op.bytes, NULL);



    avpkt->pts = ff_samples_to_time_base(avctx, op.granulepos);



    duration = avpriv_vorbis_parse_frame(&s->vp, avpkt->data, avpkt->size);

    if (duration > 0) {

        /* we do not know encoder delay until we get the first packet from

         * libvorbis, so we have to update the AudioFrameQueue counts */

        if (!avctx->delay) {

            avctx->delay              = duration;

            s->afq.remaining_delay   += duration;

            s->afq.remaining_samples += duration;

        }

        ff_af_queue_remove(&s->afq, duration, &avpkt->pts, &avpkt->duration);

    }



    *got_packet_ptr = 1;

    return 0;

}
