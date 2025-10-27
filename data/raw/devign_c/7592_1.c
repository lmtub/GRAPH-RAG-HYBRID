int ff_mpeg4_frame_end(AVCodecContext *avctx, const uint8_t *buf, int buf_size)

{

    Mpeg4DecContext *ctx = avctx->priv_data;

    MpegEncContext    *s = &ctx->m;



    /* divx 5.01+ bitstream reorder stuff */

    /* Since this clobbers the input buffer and hwaccel codecs still need the

     * data during hwaccel->end_frame we should not do this any earlier */

    if (s->divx_packed) {

        int current_pos     = s->gb.buffer == s->bitstream_buffer ? 0 : (get_bits_count(&s->gb) >> 3);

        int startcode_found = 0;



        if (buf_size - current_pos > 7) {



            int i;

            for (i = current_pos; i < buf_size - 4; i++)



                if (buf[i]     == 0 &&

                    buf[i + 1] == 0 &&

                    buf[i + 2] == 1 &&

                    buf[i + 3] == 0xB6) {

                    startcode_found = !(buf[i + 4] & 0x40);

                    break;

                }

        }



        if (startcode_found) {

            av_fast_malloc(&s->bitstream_buffer,

                           &s->allocated_bitstream_buffer_size,

                           buf_size - current_pos +

                           FF_INPUT_BUFFER_PADDING_SIZE);

            if (!s->bitstream_buffer)

                return AVERROR(ENOMEM);

            memcpy(s->bitstream_buffer, buf + current_pos,

                   buf_size - current_pos);

            s->bitstream_buffer_size = buf_size - current_pos;

        }

    }



    return 0;

}
