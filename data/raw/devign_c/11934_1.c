static int a64_write_packet(struct AVFormatContext *s, AVPacket *pkt)

{

    AVCodecContext *avctx = s->streams[0]->codec;

    A64MuxerContext *c = s->priv_data;

    int i, j;

    int ch_chunksize;

    int lifetime;

    int frame_count;

    int charset_size;

    int frame_size;

    int num_frames;



    /* fetch values from extradata */

    switch (avctx->codec->id) {

    case CODEC_ID_A64_MULTI:

    case CODEC_ID_A64_MULTI5:

        if(c->interleaved) {

            /* Write interleaved, means we insert chunks of the future charset before each current frame.

             * Reason: if we load 1 charset + corresponding frames in one block on c64, we need to store

             * them first and then display frame by frame to keep in sync. Thus we would read and write

             * the data for colram from/to ram first and waste too much time. If we interleave and send the

             * charset beforehand, we assemble a new charset chunk by chunk, write current screen data to

             * screen-ram to be displayed and decode the colram directly to colram-location $d800 during

             * the overscan, while reading directly from source

             * This is the only way so far, to achieve 25fps on c64 */

            if(avctx->extradata) {

                /* fetch values from extradata */

                lifetime     = AV_RB32(avctx->extradata + 0);

                frame_count  = AV_RB32(avctx->extradata + 4);

                charset_size = AV_RB32(avctx->extradata + 8);

                frame_size   = AV_RB32(avctx->extradata + 12);



                /* TODO: sanity checks? */

            }

            else {

                av_log(avctx, AV_LOG_ERROR, "extradata not set\n");

                return AVERROR(EINVAL);

            }

            ch_chunksize=charset_size/lifetime;

            /* TODO: check if charset/size is % lifetime, but maybe check in codec */

            if(pkt->data) num_frames = lifetime;

            else num_frames = c->prev_frame_count;

            for(i = 0; i < num_frames; i++) {

                if(pkt->data) {

                    /* if available, put newest charset chunk into buffer */

                    put_buffer(s->pb, pkt->data + ch_chunksize * i, ch_chunksize);

                }

                else {

                    /* a bit ugly, but is there an alternative to put many zeros? */

                    for(j = 0; j < ch_chunksize; j++) put_byte(s->pb, 0);

                }

                if(c->prev_pkt.data) {

                    /* put frame (screen + colram) from last packet into buffer */

                    put_buffer(s->pb, c->prev_pkt.data + charset_size + frame_size * i, frame_size);

                }

                else {

                    /* a bit ugly, but is there an alternative to put many zeros? */

                    for(j = 0; j < frame_size; j++) put_byte(s->pb, 0);

                }

            }

            /* backup current packet for next turn */

            if(pkt->data) {

                av_new_packet(&c->prev_pkt, pkt->size);

                memcpy(c->prev_pkt.data, pkt->data, pkt->size);

            }

            c->prev_frame_count = frame_count;

            break;

        }

        default:

            /* Write things as is. Nice for self-contained frames from non-multicolor modes or if played

             * directly from ram and not from a streaming device (rrnet/mmc) */

            if(pkt) put_buffer(s->pb, pkt->data, pkt->size);

        break;

    }



    put_flush_packet(s->pb);

    return 0;

}
