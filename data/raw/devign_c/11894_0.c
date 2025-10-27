static int wmavoice_decode_packet(AVCodecContext *ctx, void *data,

                                  int *data_size, AVPacket *avpkt)

{

    WMAVoiceContext *s = ctx->priv_data;

    GetBitContext *gb = &s->gb;

    int size, res, pos;



    if (*data_size < 480 * sizeof(float)) {

        av_log(ctx, AV_LOG_ERROR,

               "Output buffer too small (%d given - %zu needed)\n",

               *data_size, 480 * sizeof(float));

        return -1;

    }



    /* Packets are sometimes a multiple of ctx->block_align, with a packet

     * header at each ctx->block_align bytes. However, Libav's ASF demuxer

     * feeds us ASF packets, which may concatenate multiple "codec" packets

     * in a single "muxer" packet, so we artificially emulate that by

     * capping the packet size at ctx->block_align. */

    for (size = avpkt->size; size > ctx->block_align; size -= ctx->block_align);

    if (!size) {

        *data_size = 0;

        return 0;

    }

    init_get_bits(&s->gb, avpkt->data, size << 3);



    /* size == ctx->block_align is used to indicate whether we are dealing with

     * a new packet or a packet of which we already read the packet header

     * previously. */

    if (size == ctx->block_align) { // new packet header

        if ((res = parse_packet_header(s)) < 0)

            return res;



        /* If the packet header specifies a s->spillover_nbits, then we want

         * to push out all data of the previous packet (+ spillover) before

         * continuing to parse new superframes in the current packet. */

        if (s->spillover_nbits > 0) {

            if (s->sframe_cache_size > 0) {

                int cnt = get_bits_count(gb);

                copy_bits(&s->pb, avpkt->data, size, gb, s->spillover_nbits);

                flush_put_bits(&s->pb);

                s->sframe_cache_size += s->spillover_nbits;

                if ((res = synth_superframe(ctx, data, data_size)) == 0 &&

                    *data_size > 0) {

                    cnt += s->spillover_nbits;

                    s->skip_bits_next = cnt & 7;

                    return cnt >> 3;

                } else

                    skip_bits_long (gb, s->spillover_nbits - cnt +

                                    get_bits_count(gb)); // resync

            } else

                skip_bits_long(gb, s->spillover_nbits);  // resync

        }

    } else if (s->skip_bits_next)

        skip_bits(gb, s->skip_bits_next);



    /* Try parsing superframes in current packet */

    s->sframe_cache_size = 0;

    s->skip_bits_next = 0;

    pos = get_bits_left(gb);

    if ((res = synth_superframe(ctx, data, data_size)) < 0) {

        return res;

    } else if (*data_size > 0) {

        int cnt = get_bits_count(gb);

        s->skip_bits_next = cnt & 7;

        return cnt >> 3;

    } else if ((s->sframe_cache_size = pos) > 0) {

        /* rewind bit reader to start of last (incomplete) superframe... */

        init_get_bits(gb, avpkt->data, size << 3);

        skip_bits_long(gb, (size << 3) - pos);

        assert(get_bits_left(gb) == pos);



        /* ...and cache it for spillover in next packet */

        init_put_bits(&s->pb, s->sframe_cache, SFRAME_CACHE_MAXSIZE);

        copy_bits(&s->pb, avpkt->data, size, gb, s->sframe_cache_size);

        // FIXME bad - just copy bytes as whole and add use the

        // skip_bits_next field

    }



    return size;

}
