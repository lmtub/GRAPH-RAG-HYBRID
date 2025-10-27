int ff_mov_read_esds(AVFormatContext *fc, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    int tag, len;



    if (fc->nb_streams < 1)

        return 0;

    st = fc->streams[fc->nb_streams-1];



    get_be32(pb); /* version + flags */

    len = mp4_read_descr(fc, pb, &tag);

    if (tag == MP4ESDescrTag) {

        get_be16(pb); /* ID */

        get_byte(pb); /* priority */

    } else

        get_be16(pb); /* ID */



    len = mp4_read_descr(fc, pb, &tag);

    if (tag == MP4DecConfigDescrTag) {

        int object_type_id = get_byte(pb);

        get_byte(pb); /* stream type */

        get_be24(pb); /* buffer size db */

        get_be32(pb); /* max bitrate */

        get_be32(pb); /* avg bitrate */



        st->codec->codec_id= ff_codec_get_id(ff_mp4_obj_type, object_type_id);

        dprintf(fc, "esds object type id 0x%02x\n", object_type_id);

        len = mp4_read_descr(fc, pb, &tag);

        if (tag == MP4DecSpecificDescrTag) {

            dprintf(fc, "Specific MPEG4 header len=%d\n", len);

            if((uint64_t)len > (1<<30))

                return -1;


            st->codec->extradata = av_mallocz(len + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!st->codec->extradata)

                return AVERROR(ENOMEM);

            get_buffer(pb, st->codec->extradata, len);

            st->codec->extradata_size = len;

            if (st->codec->codec_id == CODEC_ID_AAC) {

                MPEG4AudioConfig cfg;

                ff_mpeg4audio_get_config(&cfg, st->codec->extradata,

                                         st->codec->extradata_size);

                st->codec->channels = cfg.channels;

                if (cfg.object_type == 29 && cfg.sampling_index < 3) // old mp3on4

                    st->codec->sample_rate = ff_mpa_freq_tab[cfg.sampling_index];

                else if (cfg.ext_sample_rate)

                    st->codec->sample_rate = cfg.ext_sample_rate;

                else

                    st->codec->sample_rate = cfg.sample_rate;

                dprintf(fc, "mp4a config channels %d obj %d ext obj %d "

                        "sample rate %d ext sample rate %d\n", st->codec->channels,

                        cfg.object_type, cfg.ext_object_type,

                        cfg.sample_rate, cfg.ext_sample_rate);

                if (!(st->codec->codec_id = ff_codec_get_id(mp4_audio_types,

                                                            cfg.object_type)))

                    st->codec->codec_id = CODEC_ID_AAC;

            }

        }

    }

    return 0;

}