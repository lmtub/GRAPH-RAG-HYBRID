static int vorbis_decode_frame(AVCodecContext *avccontext,

                               void *data, int *data_size,

                               AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    vorbis_context *vc = avccontext->priv_data ;

    GetBitContext *gb = &(vc->gb);

    const float *channel_ptrs[255];

    int i, len;



    if (!buf_size)

        return 0;



    av_dlog(NULL, "packet length %d \n", buf_size);



    init_get_bits(gb, buf, buf_size*8);



    len = vorbis_parse_audio_packet(vc);



    if (len <= 0) {

        *data_size = 0;

        return buf_size;

    }



    if (!vc->first_frame) {

        vc->first_frame = 1;

        *data_size = 0;

        return buf_size ;

    }



    av_dlog(NULL, "parsed %d bytes %d bits, returned %d samples (*ch*bits) \n",

            get_bits_count(gb) / 8, get_bits_count(gb) % 8, len);



    if (vc->audio_channels > 8) {

        for (i = 0; i < vc->audio_channels; i++)

            channel_ptrs[i] = vc->channel_floors + i * len;

    } else {

        for (i = 0; i < vc->audio_channels; i++)

            channel_ptrs[i] = vc->channel_floors +

                              len * ff_vorbis_channel_layout_offsets[vc->audio_channels - 1][i];

    }



    if (avccontext->sample_fmt == AV_SAMPLE_FMT_FLT)

        vc->fmt_conv.float_interleave(data, channel_ptrs, len, vc->audio_channels);

    else

        vc->fmt_conv.float_to_int16_interleave(data, channel_ptrs, len,

                                               vc->audio_channels);



    *data_size = len * vc->audio_channels *

                 av_get_bytes_per_sample(avccontext->sample_fmt);



    return buf_size ;

}
