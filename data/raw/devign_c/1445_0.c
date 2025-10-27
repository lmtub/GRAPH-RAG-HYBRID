int ff_get_wav_header(AVFormatContext *s, AVIOContext *pb,

                      AVCodecContext *codec, int size)

{

    int id;



    if (size < 14)

        return AVERROR_INVALIDDATA;



    id                 = avio_rl16(pb);

    codec->codec_type  = AVMEDIA_TYPE_AUDIO;

    codec->channels    = avio_rl16(pb);

    codec->sample_rate = avio_rl32(pb);

    codec->bit_rate    = avio_rl32(pb) * 8;

    codec->block_align = avio_rl16(pb);

    if (size == 14) {  /* We're dealing with plain vanilla WAVEFORMAT */

        codec->bits_per_coded_sample = 8;

    } else

        codec->bits_per_coded_sample = avio_rl16(pb);

    if (id == 0xFFFE) {

        codec->codec_tag = 0;

    } else {

        codec->codec_tag = id;

        codec->codec_id  = ff_wav_codec_get_id(id,

                                               codec->bits_per_coded_sample);

    }

    if (size >= 18) {  /* We're obviously dealing with WAVEFORMATEX */

        int cbSize = avio_rl16(pb); /* cbSize */

        size  -= 18;

        cbSize = FFMIN(size, cbSize);

        if (cbSize >= 22 && id == 0xfffe) { /* WAVEFORMATEXTENSIBLE */

            parse_waveformatex(pb, codec);

            cbSize -= 22;

            size   -= 22;

        }

        codec->extradata_size = cbSize;

        if (cbSize > 0) {

            av_free(codec->extradata);

            codec->extradata = av_mallocz(codec->extradata_size +

                                          FF_INPUT_BUFFER_PADDING_SIZE);

            if (!codec->extradata)

                return AVERROR(ENOMEM);

            avio_read(pb, codec->extradata, codec->extradata_size);

            size -= cbSize;

        }



        /* It is possible for the chunk to contain garbage at the end */

        if (size > 0)

            avio_skip(pb, size);

    }

    if (codec->sample_rate <= 0) {

        av_log(s, AV_LOG_ERROR,

               "Invalid sample rate: %d\n", codec->sample_rate);

        return AVERROR_INVALIDDATA;

    }

    if (codec->codec_id == AV_CODEC_ID_AAC_LATM) {

        /* Channels and sample_rate values are those prior to applying SBR

         * and/or PS. */

        codec->channels    = 0;

        codec->sample_rate = 0;

    }

    /* override bits_per_coded_sample for G.726 */

    if (codec->codec_id == AV_CODEC_ID_ADPCM_G726)

        codec->bits_per_coded_sample = codec->bit_rate / codec->sample_rate;



    return 0;

}
