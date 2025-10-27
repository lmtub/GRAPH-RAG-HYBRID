static int asf_read_stream_properties(AVFormatContext *s, int64_t size)

{

    ASFContext *asf = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;

    ASFStream *asf_st;

    ff_asf_guid g;

    enum AVMediaType type;

    int type_specific_size, sizeX;

    uint64_t total_size;

    unsigned int tag1;

    int64_t pos1, pos2, start_time;

    int test_for_ext_stream_audio, is_dvr_ms_audio=0;



    if (s->nb_streams == ASF_MAX_STREAMS) {

        av_log(s, AV_LOG_ERROR, "too many streams\n");

        return AVERROR(EINVAL);

    }



    pos1 = avio_tell(pb);



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 32, 1, 1000); /* 32 bit pts in ms */

    asf_st = av_mallocz(sizeof(ASFStream));

    if (!asf_st)

        return AVERROR(ENOMEM);

    st->priv_data = asf_st;

    st->start_time = 0;

    start_time = asf->hdr.preroll;



    asf_st->stream_language_index = 128; // invalid stream index means no language info



    if(!(asf->hdr.flags & 0x01)) { // if we aren't streaming...

        st->duration = asf->hdr.play_time /

            (10000000 / 1000) - start_time;

    }

    ff_get_guid(pb, &g);



    test_for_ext_stream_audio = 0;

    if (!ff_guidcmp(&g, &ff_asf_audio_stream)) {

        type = AVMEDIA_TYPE_AUDIO;

    } else if (!ff_guidcmp(&g, &ff_asf_video_stream)) {

        type = AVMEDIA_TYPE_VIDEO;

    } else if (!ff_guidcmp(&g, &ff_asf_jfif_media)) {

        type = AVMEDIA_TYPE_VIDEO;

        st->codec->codec_id = CODEC_ID_MJPEG;

    } else if (!ff_guidcmp(&g, &ff_asf_command_stream)) {

        type = AVMEDIA_TYPE_DATA;

    } else if (!ff_guidcmp(&g, &ff_asf_ext_stream_embed_stream_header)) {

        test_for_ext_stream_audio = 1;

        type = AVMEDIA_TYPE_UNKNOWN;

    } else {

        return -1;

    }

    ff_get_guid(pb, &g);

    total_size = avio_rl64(pb);

    type_specific_size = avio_rl32(pb);

    avio_rl32(pb);

    st->id = avio_rl16(pb) & 0x7f; /* stream id */

    // mapping of asf ID to AV stream ID;

    asf->asfid2avid[st->id] = s->nb_streams - 1;



    avio_rl32(pb);



    if (test_for_ext_stream_audio) {

        ff_get_guid(pb, &g);

        if (!ff_guidcmp(&g, &ff_asf_ext_stream_audio_stream)) {

            type = AVMEDIA_TYPE_AUDIO;

            is_dvr_ms_audio=1;

            ff_get_guid(pb, &g);

            avio_rl32(pb);

            avio_rl32(pb);

            avio_rl32(pb);

            ff_get_guid(pb, &g);

            avio_rl32(pb);

        }

    }



    st->codec->codec_type = type;

    if (type == AVMEDIA_TYPE_AUDIO) {

        ff_get_wav_header(pb, st->codec, type_specific_size);

        if (is_dvr_ms_audio) {

            // codec_id and codec_tag are unreliable in dvr_ms

            // files. Set them later by probing stream.

            st->codec->codec_id = CODEC_ID_PROBE;

            st->codec->codec_tag = 0;

        }

        if (st->codec->codec_id == CODEC_ID_AAC) {

            st->need_parsing = AVSTREAM_PARSE_NONE;

        } else {

            st->need_parsing = AVSTREAM_PARSE_FULL;

        }

        /* We have to init the frame size at some point .... */

        pos2 = avio_tell(pb);

        if (size >= (pos2 + 8 - pos1 + 24)) {

            asf_st->ds_span = avio_r8(pb);

            asf_st->ds_packet_size = avio_rl16(pb);

            asf_st->ds_chunk_size = avio_rl16(pb);

            avio_rl16(pb); //ds_data_size

            avio_r8(pb);   //ds_silence_data

        }

        //printf("Descrambling: ps:%d cs:%d ds:%d s:%d  sd:%d\n",

        //       asf_st->ds_packet_size, asf_st->ds_chunk_size,

        //       asf_st->ds_data_size, asf_st->ds_span, asf_st->ds_silence_data);

        if (asf_st->ds_span > 1) {

            if (!asf_st->ds_chunk_size

                    || (asf_st->ds_packet_size/asf_st->ds_chunk_size <= 1)

                    || asf_st->ds_packet_size % asf_st->ds_chunk_size)

                asf_st->ds_span = 0; // disable descrambling

        }

        switch (st->codec->codec_id) {

            case CODEC_ID_MP3:

                st->codec->frame_size = MPA_FRAME_SIZE;

                break;

            case CODEC_ID_PCM_S16LE:

            case CODEC_ID_PCM_S16BE:

            case CODEC_ID_PCM_U16LE:

            case CODEC_ID_PCM_U16BE:

            case CODEC_ID_PCM_S8:

            case CODEC_ID_PCM_U8:

            case CODEC_ID_PCM_ALAW:

            case CODEC_ID_PCM_MULAW:

                st->codec->frame_size = 1;

                break;

            default:

                /* This is probably wrong, but it prevents a crash later */

                st->codec->frame_size = 1;

                break;

        }

    } else if (type == AVMEDIA_TYPE_VIDEO &&

            size - (avio_tell(pb) - pos1 + 24) >= 51) {

        avio_rl32(pb);

        avio_rl32(pb);

        avio_r8(pb);

        avio_rl16(pb);        /* size */

        sizeX= avio_rl32(pb); /* size */

        st->codec->width = avio_rl32(pb);

        st->codec->height = avio_rl32(pb);

        /* not available for asf */

        avio_rl16(pb); /* panes */

        st->codec->bits_per_coded_sample = avio_rl16(pb); /* depth */

        tag1 = avio_rl32(pb);

        avio_skip(pb, 20);

        //                av_log(s, AV_LOG_DEBUG, "size:%d tsize:%d sizeX:%d\n", size, total_size, sizeX);

        if (sizeX > 40) {

            st->codec->extradata_size = sizeX - 40;

            st->codec->extradata = av_mallocz(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

            avio_read(pb, st->codec->extradata, st->codec->extradata_size);

        }



        /* Extract palette from extradata if bpp <= 8 */

        /* This code assumes that extradata contains only palette */

        /* This is true for all paletted codecs implemented in ffmpeg */

        if (st->codec->extradata_size && (st->codec->bits_per_coded_sample <= 8)) {

            int av_unused i;

            st->codec->palctrl = av_mallocz(sizeof(AVPaletteControl));

#if HAVE_BIGENDIAN

            for (i = 0; i < FFMIN(st->codec->extradata_size, AVPALETTE_SIZE)/4; i++)

                st->codec->palctrl->palette[i] = av_bswap32(((uint32_t*)st->codec->extradata)[i]);

#else

            memcpy(st->codec->palctrl->palette, st->codec->extradata,

                    FFMIN(st->codec->extradata_size, AVPALETTE_SIZE));

#endif

            st->codec->palctrl->palette_changed = 1;

        }



        st->codec->codec_tag = tag1;

        st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, tag1);

        if(tag1 == MKTAG('D', 'V', 'R', ' ')){

            st->need_parsing = AVSTREAM_PARSE_FULL;

            // issue658 containse wrong w/h and MS even puts a fake seq header with wrong w/h in extradata while a correct one is in te stream. maximum lameness

            st->codec->width  =

                st->codec->height = 0;

            av_freep(&st->codec->extradata);

            st->codec->extradata_size=0;

        }

        if(st->codec->codec_id == CODEC_ID_H264)

            st->need_parsing = AVSTREAM_PARSE_FULL_ONCE;

    }

    pos2 = avio_tell(pb);

    avio_skip(pb, size - (pos2 - pos1 + 24));



    return 0;

}
