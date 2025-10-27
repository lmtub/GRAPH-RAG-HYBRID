static void parse_waveformatex(AVIOContext *pb, AVCodecParameters *par)

{

    ff_asf_guid subformat;

    par->bits_per_coded_sample = avio_rl16(pb);

    par->channel_layout        = avio_rl32(pb); /* dwChannelMask */



    ff_get_guid(pb, &subformat);

    if (!memcmp(subformat + 4,

                (const uint8_t[]){ FF_MEDIASUBTYPE_BASE_GUID }, 12)) {

        par->codec_tag = AV_RL32(subformat);

        par->codec_id  = ff_wav_codec_get_id(par->codec_tag,

                                             par->bits_per_coded_sample);

    } else {

        par->codec_id = ff_codec_guid_get_id(ff_codec_wav_guids, subformat);

        if (!par->codec_id)

            av_log(pb, AV_LOG_WARNING,

                   "unknown subformat:"FF_PRI_GUID"\n",

                   FF_ARG_GUID(subformat));

    }

}
