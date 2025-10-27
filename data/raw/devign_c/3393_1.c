void ff_get_wav_header(AVIOContext *pb, AVCodecContext *codec, int size)

{

    int id;



    id = avio_rl16(pb);

    codec->codec_type = AVMEDIA_TYPE_AUDIO;

    codec->codec_tag = id;

    codec->channels = avio_rl16(pb);

    codec->sample_rate = avio_rl32(pb);

    codec->bit_rate = avio_rl32(pb) * 8;

    codec->block_align = avio_rl16(pb);

    if (size == 14) {  /* We're dealing with plain vanilla WAVEFORMAT */

        codec->bits_per_coded_sample = 8;

    }else

        codec->bits_per_coded_sample = avio_rl16(pb);

    if (size >= 18) {  /* We're obviously dealing with WAVEFORMATEX */

        int cbSize = avio_rl16(pb); /* cbSize */

        size -= 18;

        cbSize = FFMIN(size, cbSize);

        if (cbSize >= 22 && id == 0xfffe) { /* WAVEFORMATEXTENSIBLE */

            codec->bits_per_coded_sample = avio_rl16(pb);

            codec->channel_layout = avio_rl32(pb); /* dwChannelMask */

            id = avio_rl32(pb); /* 4 first bytes of GUID */

            avio_skip(pb, 12); /* skip end of GUID */

            cbSize -= 22;

            size -= 22;

        }

        codec->extradata_size = cbSize;

        if (cbSize > 0) {

            codec->extradata = av_mallocz(codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

            avio_read(pb, codec->extradata, codec->extradata_size);

            size -= cbSize;

        }



        /* It is possible for the chunk to contain garbage at the end */

        if (size > 0)

            avio_skip(pb, size);

    }

    codec->codec_id = ff_wav_codec_get_id(id, codec->bits_per_coded_sample);

    if (codec->codec_id == CODEC_ID_AAC_LATM) {

        /* channels and sample_rate values are those prior to applying SBR and/or PS */

        codec->channels    = 0;

        codec->sample_rate = 0;

    }

}
