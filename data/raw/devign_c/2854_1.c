static int vorbis_decode_frame(AVCodecContext *avctx, void *data,

                               int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    vorbis_context *vc = avctx->priv_data;

    AVFrame *frame     = data;

    GetBitContext *gb = &vc->gb;

    float *channel_ptrs[255];

    int i, len, ret;



    av_dlog(NULL, "packet length %d \n", buf_size);



    /* get output buffer */

    frame->nb_samples = vc->blocksize[1] / 2;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    if (vc->audio_channels > 8) {

        for (i = 0; i < vc->audio_channels; i++)

            channel_ptrs[i] = (float *)frame->extended_data[i];

    } else {

        for (i = 0; i < vc->audio_channels; i++) {

            int ch = ff_vorbis_channel_layout_offsets[vc->audio_channels - 1][i];

            channel_ptrs[ch] = (float *)frame->extended_data[i];

        }

    }



    init_get_bits(gb, buf, buf_size*8);



    if ((len = vorbis_parse_audio_packet(vc, channel_ptrs)) <= 0)

        return len;



    if (!vc->first_frame) {

        vc->first_frame = 1;

        *got_frame_ptr = 0;


        return buf_size;

    }



    av_dlog(NULL, "parsed %d bytes %d bits, returned %d samples (*ch*bits) \n",

            get_bits_count(gb) / 8, get_bits_count(gb) % 8, len);



    frame->nb_samples = len;

    *got_frame_ptr    = 1;



    return buf_size;

}