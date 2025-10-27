static int amf_parse_object(AVFormatContext *s, AVStream *astream,

                            AVStream *vstream, const char *key,

                            int64_t max_pos, int depth)

{

    AVCodecContext *acodec, *vcodec;

    FLVContext *flv = s->priv_data;

    AVIOContext *ioc;

    AMFDataType amf_type;

    char str_val[256];

    double num_val;



    num_val  = 0;

    ioc      = s->pb;

    amf_type = avio_r8(ioc);



    switch (amf_type) {

    case AMF_DATA_TYPE_NUMBER:

        num_val = av_int2double(avio_rb64(ioc));

        break;

    case AMF_DATA_TYPE_BOOL:

        num_val = avio_r8(ioc);

        break;

    case AMF_DATA_TYPE_STRING:

        if (amf_get_string(ioc, str_val, sizeof(str_val)) < 0)

            return -1;

        break;

    case AMF_DATA_TYPE_OBJECT:

        if ((vstream || astream) && key &&

            ioc->seekable &&

            !strcmp(KEYFRAMES_TAG, key) && depth == 1)

            if (parse_keyframes_index(s, ioc, vstream ? vstream : astream,

                                      max_pos) < 0)

                av_log(s, AV_LOG_ERROR, "Keyframe index parsing failed\n");



        while (avio_tell(ioc) < max_pos - 2 &&

               amf_get_string(ioc, str_val, sizeof(str_val)) > 0)

            if (amf_parse_object(s, astream, vstream, str_val, max_pos,

                                 depth + 1) < 0)

                return -1;     // if we couldn't skip, bomb out.

        if (avio_r8(ioc) != AMF_END_OF_OBJECT)

            return -1;

        break;

    case AMF_DATA_TYPE_NULL:

    case AMF_DATA_TYPE_UNDEFINED:

    case AMF_DATA_TYPE_UNSUPPORTED:

        break;     // these take up no additional space

    case AMF_DATA_TYPE_MIXEDARRAY:

        avio_skip(ioc, 4);     // skip 32-bit max array index

        while (avio_tell(ioc) < max_pos - 2 &&

               amf_get_string(ioc, str_val, sizeof(str_val)) > 0)

            // this is the only case in which we would want a nested

            // parse to not skip over the object

            if (amf_parse_object(s, astream, vstream, str_val, max_pos,

                                 depth + 1) < 0)

                return -1;

        if (avio_r8(ioc) != AMF_END_OF_OBJECT)

            return -1;

        break;

    case AMF_DATA_TYPE_ARRAY:

    {

        unsigned int arraylen, i;



        arraylen = avio_rb32(ioc);

        for (i = 0; i < arraylen && avio_tell(ioc) < max_pos - 1; i++)

            if (amf_parse_object(s, NULL, NULL, NULL, max_pos,

                                 depth + 1) < 0)

                return -1;      // if we couldn't skip, bomb out.

    }

    break;

    case AMF_DATA_TYPE_DATE:

        avio_skip(ioc, 8 + 2);  // timestamp (double) and UTC offset (int16)

        break;

    default:                    // unsupported type, we couldn't skip

        return -1;

    }



    if (key) {

        // stream info doesn't live any deeper than the first object

        if (depth == 1) {

            acodec = astream ? astream->codec : NULL;

            vcodec = vstream ? vstream->codec : NULL;



            if (amf_type == AMF_DATA_TYPE_NUMBER ||

                amf_type == AMF_DATA_TYPE_BOOL) {

                if (!strcmp(key, "duration"))

                    s->duration = num_val * AV_TIME_BASE;

                else if (!strcmp(key, "videodatarate") && vcodec &&

                         0 <= (int)(num_val * 1024.0))

                    vcodec->bit_rate = num_val * 1024.0;

                else if (!strcmp(key, "audiodatarate") && acodec &&

                         0 <= (int)(num_val * 1024.0))

                    acodec->bit_rate = num_val * 1024.0;

                else if (!strcmp(key, "datastream")) {

                    AVStream *st = create_stream(s, AVMEDIA_TYPE_DATA);

                    if (!st)

                        return AVERROR(ENOMEM);

                    st->codec->codec_id = AV_CODEC_ID_TEXT;

                } else if (flv->trust_metadata) {

                    if (!strcmp(key, "videocodecid") && vcodec) {

                        flv_set_video_codec(s, vstream, num_val, 0);

                    } else if (!strcmp(key, "audiocodecid") && acodec) {

                        int id = ((int)num_val) << FLV_AUDIO_CODECID_OFFSET;

                        flv_set_audio_codec(s, astream, acodec, id);

                    } else if (!strcmp(key, "audiosamplerate") && acodec) {

                        acodec->sample_rate = num_val;

                    } else if (!strcmp(key, "audiosamplesize") && acodec) {

                        acodec->bits_per_coded_sample = num_val;

                    } else if (!strcmp(key, "stereo") && acodec) {

                        acodec->channels       = num_val + 1;

                        acodec->channel_layout = acodec->channels == 2 ?

                                                 AV_CH_LAYOUT_STEREO :

                                                 AV_CH_LAYOUT_MONO;

                    } else if (!strcmp(key, "width") && vcodec) {

                        vcodec->width = num_val;

                    } else if (!strcmp(key, "height") && vcodec) {

                        vcodec->height = num_val;

                    }

                }

            }

        }



        if (amf_type == AMF_DATA_TYPE_OBJECT && s->nb_streams == 1 &&

           ((!acodec && !strcmp(key, "audiocodecid")) ||

            (!vcodec && !strcmp(key, "videocodecid"))))

                s->ctx_flags &= ~AVFMTCTX_NOHEADER; //If there is either audio/video missing, codecid will be an empty object



        if (!strcmp(key, "duration")        ||

            !strcmp(key, "filesize")        ||

            !strcmp(key, "width")           ||

            !strcmp(key, "height")          ||

            !strcmp(key, "videodatarate")   ||

            !strcmp(key, "framerate")       ||

            !strcmp(key, "videocodecid")    ||

            !strcmp(key, "audiodatarate")   ||

            !strcmp(key, "audiosamplerate") ||

            !strcmp(key, "audiosamplesize") ||

            !strcmp(key, "stereo")          ||

            !strcmp(key, "audiocodecid")    ||

            !strcmp(key, "datastream"))

            return 0;



        s->event_flags |= AVFMT_EVENT_FLAG_METADATA_UPDATED;

        if (amf_type == AMF_DATA_TYPE_BOOL) {

            av_strlcpy(str_val, num_val > 0 ? "true" : "false",

                       sizeof(str_val));

            av_dict_set(&s->metadata, key, str_val, 0);

        } else if (amf_type == AMF_DATA_TYPE_NUMBER) {

            snprintf(str_val, sizeof(str_val), "%.f", num_val);

            av_dict_set(&s->metadata, key, str_val, 0);

        } else if (amf_type == AMF_DATA_TYPE_STRING)

            av_dict_set(&s->metadata, key, str_val, 0);

    }



    return 0;

}
