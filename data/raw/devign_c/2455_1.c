static int flv_write_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    FLVContext *flv = s->priv_data;

    AVCodecContext *audio_enc = NULL, *video_enc = NULL, *data_enc = NULL;

    int i, metadata_count = 0;

    double framerate = 0.0;

    int64_t metadata_size_pos, data_size, metadata_count_pos;

    AVDictionaryEntry *tag = NULL;



    for (i = 0; i < s->nb_streams; i++) {

        AVCodecContext *enc = s->streams[i]->codec;

        FLVStreamContext *sc;

        switch (enc->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            if (s->streams[i]->r_frame_rate.den &&

                s->streams[i]->r_frame_rate.num) {

                framerate = av_q2d(s->streams[i]->r_frame_rate);

            } else {

                framerate = 1 / av_q2d(s->streams[i]->codec->time_base);

            }

            video_enc = enc;

            if (enc->codec_tag == 0) {

                av_log(s, AV_LOG_ERROR, "video codec not compatible with flv\n");

                return -1;

            }

            break;

        case AVMEDIA_TYPE_AUDIO:

            audio_enc = enc;

            if (get_audio_flags(s, enc) < 0)

                return AVERROR_INVALIDDATA;

            break;

        case AVMEDIA_TYPE_DATA:

            if (enc->codec_id != CODEC_ID_TEXT) {

                av_log(s, AV_LOG_ERROR, "codec not compatible with flv\n");

                return AVERROR_INVALIDDATA;

            }

            data_enc = enc;

            break;

        default:

            av_log(s, AV_LOG_ERROR, "codec not compatible with flv\n");

            return -1;

        }

        avpriv_set_pts_info(s->streams[i], 32, 1, 1000); /* 32 bit pts in ms */



        sc = av_mallocz(sizeof(FLVStreamContext));

        if (!sc)

            return AVERROR(ENOMEM);

        s->streams[i]->priv_data = sc;

        sc->last_ts = -1;

    }



    flv->delay = AV_NOPTS_VALUE;



    avio_write(pb, "FLV", 3);

    avio_w8(pb, 1);

    avio_w8(pb, FLV_HEADER_FLAG_HASAUDIO * !!audio_enc +

                FLV_HEADER_FLAG_HASVIDEO * !!video_enc);

    avio_wb32(pb, 9);

    avio_wb32(pb, 0);



    for (i = 0; i < s->nb_streams; i++)

        if (s->streams[i]->codec->codec_tag == 5) {

            avio_w8(pb, 8);     // message type

            avio_wb24(pb, 0);   // include flags

            avio_wb24(pb, 0);   // time stamp

            avio_wb32(pb, 0);   // reserved

            avio_wb32(pb, 11);  // size

            flv->reserved = 5;

        }



    /* write meta_tag */

    avio_w8(pb, 18);            // tag type META

    metadata_size_pos = avio_tell(pb);

    avio_wb24(pb, 0);           // size of data part (sum of all parts below)

    avio_wb24(pb, 0);           // timestamp

    avio_wb32(pb, 0);           // reserved



    /* now data of data_size size */



    /* first event name as a string */

    avio_w8(pb, AMF_DATA_TYPE_STRING);

    put_amf_string(pb, "onMetaData"); // 12 bytes



    /* mixed array (hash) with size and string/type/data tuples */

    avio_w8(pb, AMF_DATA_TYPE_MIXEDARRAY);

    metadata_count_pos = avio_tell(pb);

    metadata_count = 5 * !!video_enc +

                     5 * !!audio_enc +

                     1 * !!data_enc  +

                     2; // +2 for duration and file size



    avio_wb32(pb, metadata_count);



    put_amf_string(pb, "duration");

    flv->duration_offset= avio_tell(pb);



    // fill in the guessed duration, it'll be corrected later if incorrect

    put_amf_double(pb, s->duration / AV_TIME_BASE);



    if (video_enc) {

        put_amf_string(pb, "width");

        put_amf_double(pb, video_enc->width);



        put_amf_string(pb, "height");

        put_amf_double(pb, video_enc->height);



        put_amf_string(pb, "videodatarate");

        put_amf_double(pb, video_enc->bit_rate / 1024.0);



        put_amf_string(pb, "framerate");

        put_amf_double(pb, framerate);



        put_amf_string(pb, "videocodecid");

        put_amf_double(pb, video_enc->codec_tag);

    }



    if (audio_enc) {

        put_amf_string(pb, "audiodatarate");

        put_amf_double(pb, audio_enc->bit_rate / 1024.0);



        put_amf_string(pb, "audiosamplerate");

        put_amf_double(pb, audio_enc->sample_rate);



        put_amf_string(pb, "audiosamplesize");

        put_amf_double(pb, audio_enc->codec_id == CODEC_ID_PCM_U8 ? 8 : 16);



        put_amf_string(pb, "stereo");

        put_amf_bool(pb, audio_enc->channels == 2);



        put_amf_string(pb, "audiocodecid");

        put_amf_double(pb, audio_enc->codec_tag);

    }



    if (data_enc) {

        put_amf_string(pb, "datastream");

        put_amf_double(pb, 0.0);

    }



    while ((tag = av_dict_get(s->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {

        put_amf_string(pb, tag->key);

        avio_w8(pb, AMF_DATA_TYPE_STRING);

        put_amf_string(pb, tag->value);

        metadata_count++;

    }



    put_amf_string(pb, "filesize");

    flv->filesize_offset = avio_tell(pb);

    put_amf_double(pb, 0); // delayed write



    put_amf_string(pb, "");

    avio_w8(pb, AMF_END_OF_OBJECT);



    /* write total size of tag */

    data_size = avio_tell(pb) - metadata_size_pos - 10;



    avio_seek(pb, metadata_count_pos, SEEK_SET);

    avio_wb32(pb, metadata_count);



    avio_seek(pb, metadata_size_pos, SEEK_SET);

    avio_wb24(pb, data_size);

    avio_skip(pb, data_size + 10 - 3);

    avio_wb32(pb, data_size + 11);



    for (i = 0; i < s->nb_streams; i++) {

        AVCodecContext *enc = s->streams[i]->codec;

        if (enc->codec_id == CODEC_ID_AAC || enc->codec_id == CODEC_ID_H264) {

            int64_t pos;

            avio_w8(pb, enc->codec_type == AVMEDIA_TYPE_VIDEO ?

                    FLV_TAG_TYPE_VIDEO : FLV_TAG_TYPE_AUDIO);

            avio_wb24(pb, 0); // size patched later

            avio_wb24(pb, 0); // ts

            avio_w8(pb, 0);   // ts ext

            avio_wb24(pb, 0); // streamid

            pos = avio_tell(pb);

            if (enc->codec_id == CODEC_ID_AAC) {

                avio_w8(pb, get_audio_flags(s, enc));

                avio_w8(pb, 0); // AAC sequence header

                avio_write(pb, enc->extradata, enc->extradata_size);

            } else {

                avio_w8(pb, enc->codec_tag | FLV_FRAME_KEY); // flags

                avio_w8(pb, 0); // AVC sequence header

                avio_wb24(pb, 0); // composition time

                ff_isom_write_avcc(pb, enc->extradata, enc->extradata_size);

            }

            data_size = avio_tell(pb) - pos;

            avio_seek(pb, -data_size - 10, SEEK_CUR);

            avio_wb24(pb, data_size);

            avio_skip(pb, data_size + 10 - 3);

            avio_wb32(pb, data_size + 11); // previous tag size

        }

    }



    return 0;

}
