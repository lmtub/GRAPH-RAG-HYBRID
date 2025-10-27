static int oggvorbis_encode_frame(AVCodecContext *avctx, unsigned char *packets,

                                  int buf_size, void *data)

{

    OggVorbisContext *s = avctx->priv_data;

    ogg_packet op;

    float *audio = data;

    int pkt_size;



    /* send samples to libvorbis */

    if (data) {

        const int samples = avctx->frame_size;

        float **buffer;

        int c, channels = s->vi.channels;



        buffer = vorbis_analysis_buffer(&s->vd, samples);

        for (c = 0; c < channels; c++) {

            int i;

            int co = (channels > 8) ? c :

                     ff_vorbis_encoding_channel_layout_offsets[channels - 1][c];

            for (i = 0; i < samples; i++)

                buffer[c][i] = audio[i * channels + co];

        }

        vorbis_analysis_wrote(&s->vd, samples);

    } else {

        if (!s->eof)

            vorbis_analysis_wrote(&s->vd, 0);

        s->eof = 1;

    }



    /* retrieve available packets from libvorbis */

    while (vorbis_analysis_blockout(&s->vd, &s->vb) == 1) {

        vorbis_analysis(&s->vb, NULL);

        vorbis_bitrate_addblock(&s->vb);



        /* add any available packets to the output packet buffer */

        while (vorbis_bitrate_flushpacket(&s->vd, &op)) {

            /* i'd love to say the following line is a hack, but sadly it's

             * not, apparently the end of stream decision is in libogg. */

            if (op.bytes == 1 && op.e_o_s)

                continue;

            if (s->buffer_index + sizeof(ogg_packet) + op.bytes > BUFFER_SIZE) {

                av_log(avctx, AV_LOG_ERROR, "libvorbis: buffer overflow.");

                return -1;

            }

            memcpy(s->buffer + s->buffer_index, &op, sizeof(ogg_packet));

            s->buffer_index += sizeof(ogg_packet);

            memcpy(s->buffer + s->buffer_index, op.packet, op.bytes);

            s->buffer_index += op.bytes;

        }

    }



    /* output then next packet from the output buffer, if available */

    pkt_size = 0;

    if (s->buffer_index) {

        ogg_packet *op2 = (ogg_packet *)s->buffer;

        op2->packet     = s->buffer + sizeof(ogg_packet);



        pkt_size = op2->bytes;

        // FIXME: we should use the user-supplied pts and duration

        avctx->coded_frame->pts = ff_samples_to_time_base(avctx,

                                                          op2->granulepos);

        if (pkt_size > buf_size) {

            av_log(avctx, AV_LOG_ERROR, "libvorbis: buffer overflow.");

            return -1;

        }



        memcpy(packets, op2->packet, pkt_size);

        s->buffer_index -= pkt_size + sizeof(ogg_packet);

        memmove(s->buffer, s->buffer + pkt_size + sizeof(ogg_packet),

                s->buffer_index);

    }



    return pkt_size;

}
