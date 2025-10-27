static void mov_parse_stsd_audio(MOVContext *c, AVIOContext *pb,

                                 AVStream *st, MOVStreamContext *sc)

{

    int bits_per_sample, flags;

    uint16_t version = avio_rb16(pb);

    AVDictionaryEntry *compatible_brands = av_dict_get(c->fc->metadata, "compatible_brands", NULL, AV_DICT_MATCH_CASE);



    avio_rb16(pb); /* revision level */

    avio_rb32(pb); /* vendor */



    st->codec->channels              = avio_rb16(pb); /* channel count */

    st->codec->bits_per_coded_sample = avio_rb16(pb); /* sample size */

    av_log(c->fc, AV_LOG_TRACE, "audio channels %d\n", st->codec->channels);



    sc->audio_cid = avio_rb16(pb);

    avio_rb16(pb); /* packet size = 0 */



    st->codec->sample_rate = ((avio_rb32(pb) >> 16));



    // Read QT version 1 fields. In version 0 these do not exist.

    av_log(c->fc, AV_LOG_TRACE, "version =%d, isom =%d\n", version, c->isom);

    if (!c->isom ||

        (compatible_brands && strstr(compatible_brands->value, "qt  "))) {



        if (version == 1) {

            sc->samples_per_frame = avio_rb32(pb);

            avio_rb32(pb); /* bytes per packet */

            sc->bytes_per_frame = avio_rb32(pb);

            avio_rb32(pb); /* bytes per sample */

        } else if (version == 2) {

            avio_rb32(pb); /* sizeof struct only */

            st->codec->sample_rate = av_int2double(avio_rb64(pb));

            st->codec->channels    = avio_rb32(pb);

            avio_rb32(pb); /* always 0x7F000000 */

            st->codec->bits_per_coded_sample = avio_rb32(pb);



            flags = avio_rb32(pb); /* lpcm format specific flag */

            sc->bytes_per_frame   = avio_rb32(pb);

            sc->samples_per_frame = avio_rb32(pb);

            if (st->codec->codec_tag == MKTAG('l','p','c','m'))

                st->codec->codec_id =

                    ff_mov_get_lpcm_codec_id(st->codec->bits_per_coded_sample,

                                             flags);

        }

        if (version == 0 || (version == 1 && sc->audio_cid != -2)) {

            /* can't correctly handle variable sized packet as audio unit */

            switch (st->codec->codec_id) {

            case AV_CODEC_ID_MP2:

            case AV_CODEC_ID_MP3:

                st->need_parsing = AVSTREAM_PARSE_FULL;

                break;

            }

        }

    }



    if (sc->format == 0) {

        if (st->codec->bits_per_coded_sample == 8)

            st->codec->codec_id = mov_codec_id(st, MKTAG('r','a','w',' '));

        else if (st->codec->bits_per_coded_sample == 16)

            st->codec->codec_id = mov_codec_id(st, MKTAG('t','w','o','s'));

    }



    switch (st->codec->codec_id) {

    case AV_CODEC_ID_PCM_S8:

    case AV_CODEC_ID_PCM_U8:

        if (st->codec->bits_per_coded_sample == 16)

            st->codec->codec_id = AV_CODEC_ID_PCM_S16BE;

        break;

    case AV_CODEC_ID_PCM_S16LE:

    case AV_CODEC_ID_PCM_S16BE:

        if (st->codec->bits_per_coded_sample == 8)

            st->codec->codec_id = AV_CODEC_ID_PCM_S8;

        else if (st->codec->bits_per_coded_sample == 24)

            st->codec->codec_id =

                st->codec->codec_id == AV_CODEC_ID_PCM_S16BE ?

                AV_CODEC_ID_PCM_S24BE : AV_CODEC_ID_PCM_S24LE;

        else if (st->codec->bits_per_coded_sample == 32)

             st->codec->codec_id =

                st->codec->codec_id == AV_CODEC_ID_PCM_S16BE ?

                AV_CODEC_ID_PCM_S32BE : AV_CODEC_ID_PCM_S32LE;

        break;

    /* set values for old format before stsd version 1 appeared */

    case AV_CODEC_ID_MACE3:

        sc->samples_per_frame = 6;

        sc->bytes_per_frame   = 2 * st->codec->channels;

        break;

    case AV_CODEC_ID_MACE6:

        sc->samples_per_frame = 6;

        sc->bytes_per_frame   = 1 * st->codec->channels;

        break;

    case AV_CODEC_ID_ADPCM_IMA_QT:

        sc->samples_per_frame = 64;

        sc->bytes_per_frame   = 34 * st->codec->channels;

        break;

    case AV_CODEC_ID_GSM:

        sc->samples_per_frame = 160;

        sc->bytes_per_frame   = 33;

        break;

    default:

        break;

    }



    bits_per_sample = av_get_bits_per_sample(st->codec->codec_id);

    if (bits_per_sample) {

        st->codec->bits_per_coded_sample = bits_per_sample;

        sc->sample_size = (bits_per_sample >> 3) * st->codec->channels;

    }

}
