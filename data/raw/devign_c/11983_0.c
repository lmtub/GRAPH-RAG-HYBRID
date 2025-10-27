static int decode_stream_header(NUTContext *nut){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    StreamContext *stc;

    int class, stream_id;

    uint64_t tmp, end;

    AVStream *st;



    end= get_packetheader(nut, bc, 1);

    end += url_ftell(bc);



    GET_V(stream_id, tmp < s->nb_streams && !nut->stream[tmp].time_base);

    stc= &nut->stream[stream_id];



    st = s->streams[stream_id];

    if (!st)

        return AVERROR(ENOMEM);



    class = get_v(bc);

    tmp = get_fourcc(bc);

    st->codec->codec_tag= tmp;

    switch(class)

    {

        case 0:

            st->codec->codec_type = CODEC_TYPE_VIDEO;

            st->codec->codec_id = codec_get_id(codec_bmp_tags, tmp);

            if (st->codec->codec_id == CODEC_ID_NONE)

                av_log(s, AV_LOG_ERROR, "Unknown codec?!\n");

            break;

        case 1:

            st->codec->codec_type = CODEC_TYPE_AUDIO;

            st->codec->codec_id = codec_get_id(codec_wav_tags, tmp);

            if (st->codec->codec_id == CODEC_ID_NONE)

                av_log(s, AV_LOG_ERROR, "Unknown codec?!\n");

            break;

        case 2:

//            st->codec->codec_type = CODEC_TYPE_TEXT;

//            break;

        case 3:

            st->codec->codec_type = CODEC_TYPE_DATA;

            break;

        default:

            av_log(s, AV_LOG_ERROR, "Unknown stream class (%d)\n", class);

            return -1;

    }

    GET_V(stc->time_base_id    , tmp < nut->time_base_count);

    GET_V(stc->msb_pts_shift   , tmp < 16);

    stc->max_pts_distance= get_v(bc);

    GET_V(stc->decode_delay    , tmp < 1000); //sanity limit, raise this if moors law is true

    st->codec->has_b_frames= stc->decode_delay;

    get_v(bc); //stream flags



    GET_V(st->codec->extradata_size, tmp < (1<<30));

    if(st->codec->extradata_size){

        st->codec->extradata= av_mallocz(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

        get_buffer(bc, st->codec->extradata, st->codec->extradata_size);

    }



    if (st->codec->codec_type == CODEC_TYPE_VIDEO){

        GET_V(st->codec->width , tmp > 0)

        GET_V(st->codec->height, tmp > 0)

        st->codec->sample_aspect_ratio.num= get_v(bc);

        st->codec->sample_aspect_ratio.den= get_v(bc);

        if((!st->codec->sample_aspect_ratio.num) != (!st->codec->sample_aspect_ratio.den)){

            av_log(s, AV_LOG_ERROR, "invalid aspect ratio\n");

            return -1;

        }

        get_v(bc); /* csp type */

    }else if (st->codec->codec_type == CODEC_TYPE_AUDIO){

        GET_V(st->codec->sample_rate , tmp > 0)

        tmp= get_v(bc); // samplerate_den

        if(tmp > st->codec->sample_rate){

            av_log(s, AV_LOG_ERROR, "bleh, libnut muxed this ;)\n");

            st->codec->sample_rate= tmp;

        }

        GET_V(st->codec->channels, tmp > 0)

    }

    if(skip_reserved(bc, end) || get_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "Stream header %d checksum mismatch\n", stream_id);

        return -1;

    }

    stc->time_base= &nut->time_base[stc->time_base_id];

    av_set_pts_info(s->streams[stream_id], 63, stc->time_base->num, stc->time_base->den);

    return 0;

}
