static int flv_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    FLVContext *flv = s->priv_data;

    int ret, i, type, size, flags;

    int stream_type=-1;

    int64_t next, pos;

    int64_t dts, pts = AV_NOPTS_VALUE;

    AVStream *st = NULL;



 for(;;avio_skip(s->pb, 4)){ /* pkt size is repeated at end. skip it */

    pos = avio_tell(s->pb);

    type = avio_r8(s->pb);

    size = avio_rb24(s->pb);

    dts = avio_rb24(s->pb);

    dts |= avio_r8(s->pb) << 24;

    av_dlog(s, "type:%d, size:%d, dts:%"PRId64"\n", type, size, dts);

    if (url_feof(s->pb))

        return AVERROR_EOF;

    avio_skip(s->pb, 3); /* stream id, always 0 */

    flags = 0;



    if(size == 0)

        continue;



    next= size + avio_tell(s->pb);



    if (type == FLV_TAG_TYPE_AUDIO) {

        stream_type=FLV_STREAM_TYPE_AUDIO;

        flags = avio_r8(s->pb);

        size--;

    } else if (type == FLV_TAG_TYPE_VIDEO) {

        stream_type=FLV_STREAM_TYPE_VIDEO;

        flags = avio_r8(s->pb);

        size--;

        if ((flags & 0xf0) == 0x50) /* video info / command frame */

            goto skip;

    } else if (type == FLV_TAG_TYPE_META) {

        if (size > 13+1+4 && dts == 0) { // Header-type metadata stuff

            flv_read_metabody(s, next);

            goto skip;

        } else if (dts != 0) { // Script-data "special" metadata frames - don't skip

            stream_type=FLV_STREAM_TYPE_DATA;

        } else {

            goto skip;

        }

    } else {

        av_log(s, AV_LOG_DEBUG, "skipping flv packet: type %d, size %d, flags %d\n", type, size, flags);

    skip:

        avio_seek(s->pb, next, SEEK_SET);

        continue;

    }



    /* skip empty data packets */

    if (!size)

        continue;



    /* now find stream */

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        if (st->id == stream_type)

            break;

    }

    if(i == s->nb_streams){

        av_log(s, AV_LOG_WARNING, "Stream discovered after head already parsed\n");

        st= create_stream(s, stream_type);

        s->ctx_flags &= ~AVFMTCTX_NOHEADER;

    }

    av_dlog(s, "%d %X %d \n", stream_type, flags, st->discard);

    if(  (st->discard >= AVDISCARD_NONKEY && !((flags & FLV_VIDEO_FRAMETYPE_MASK) == FLV_FRAME_KEY || (stream_type == FLV_STREAM_TYPE_AUDIO)))

       ||(st->discard >= AVDISCARD_BIDIR  &&  ((flags & FLV_VIDEO_FRAMETYPE_MASK) == FLV_FRAME_DISP_INTER && (stream_type == FLV_STREAM_TYPE_VIDEO)))

       || st->discard >= AVDISCARD_ALL

       ){

        avio_seek(s->pb, next, SEEK_SET);

        continue;

    }

    if ((flags & FLV_VIDEO_FRAMETYPE_MASK) == FLV_FRAME_KEY)

        av_add_index_entry(st, pos, dts, size, 0, AVINDEX_KEYFRAME);

    break;

 }



    // if not streamed and no duration from metadata then seek to end to find the duration from the timestamps

    if(s->pb->seekable && (!s->duration || s->duration==AV_NOPTS_VALUE)){

        int size;

        const int64_t pos= avio_tell(s->pb);

        const int64_t fsize= avio_size(s->pb);

        avio_seek(s->pb, fsize-4, SEEK_SET);

        size= avio_rb32(s->pb);

        avio_seek(s->pb, fsize-3-size, SEEK_SET);

        if(size == avio_rb24(s->pb) + 11){

            uint32_t ts = avio_rb24(s->pb);

            ts |= avio_r8(s->pb) << 24;

            s->duration = ts * (int64_t)AV_TIME_BASE / 1000;

        }

        avio_seek(s->pb, pos, SEEK_SET);

    }



    if(stream_type == FLV_STREAM_TYPE_AUDIO){

        if(!st->codec->channels || !st->codec->sample_rate || !st->codec->bits_per_coded_sample) {

            st->codec->channels = (flags & FLV_AUDIO_CHANNEL_MASK) == FLV_STEREO ? 2 : 1;

            st->codec->sample_rate = (44100 << ((flags & FLV_AUDIO_SAMPLERATE_MASK) >> FLV_AUDIO_SAMPLERATE_OFFSET) >> 3);

            st->codec->bits_per_coded_sample = (flags & FLV_AUDIO_SAMPLESIZE_MASK) ? 16 : 8;

        }

        if(!st->codec->codec_id){

            flv_set_audio_codec(s, st, flags & FLV_AUDIO_CODECID_MASK);

        }

    } else if(stream_type == FLV_STREAM_TYPE_VIDEO) {

        size -= flv_set_video_codec(s, st, flags & FLV_VIDEO_CODECID_MASK);

    }



    if (st->codec->codec_id == CODEC_ID_AAC ||

        st->codec->codec_id == CODEC_ID_H264 ||

        st->codec->codec_id == CODEC_ID_MPEG4) {

        int type = avio_r8(s->pb);

        size--;

        if (st->codec->codec_id == CODEC_ID_H264 || st->codec->codec_id == CODEC_ID_MPEG4) {

            int32_t cts = (avio_rb24(s->pb)+0xff800000)^0xff800000; // sign extension

            pts = dts + cts;

            if (cts < 0) { // dts are wrong

                flv->wrong_dts = 1;

                av_log(s, AV_LOG_WARNING, "negative cts, previous timestamps might be wrong\n");

            }

            if (flv->wrong_dts)

                dts = AV_NOPTS_VALUE;

        }

        if (type == 0) {

            if ((ret = flv_get_extradata(s, st, size)) < 0)

                return ret;

            if (st->codec->codec_id == CODEC_ID_AAC) {

                MPEG4AudioConfig cfg;

                ff_mpeg4audio_get_config(&cfg, st->codec->extradata,

                                         st->codec->extradata_size);

                st->codec->channels = cfg.channels;

                if (cfg.ext_sample_rate)

                    st->codec->sample_rate = cfg.ext_sample_rate;

                else

                    st->codec->sample_rate = cfg.sample_rate;

                av_dlog(s, "mp4a config channels %d sample rate %d\n",

                        st->codec->channels, st->codec->sample_rate);

            }



            ret = AVERROR(EAGAIN);

            goto leave;

        }

    }



    /* skip empty data packets */

    if (!size) {

        ret = AVERROR(EAGAIN);

        goto leave;

    }



    ret= av_get_packet(s->pb, pkt, size);

    if (ret < 0) {

        return AVERROR(EIO);

    }

    /* note: we need to modify the packet size here to handle the last

       packet */

    pkt->size = ret;

    pkt->dts = dts;

    pkt->pts = pts == AV_NOPTS_VALUE ? dts : pts;

    pkt->stream_index = st->index;



    if (    stream_type == FLV_STREAM_TYPE_AUDIO ||

            ((flags & FLV_VIDEO_FRAMETYPE_MASK) == FLV_FRAME_KEY) ||

            stream_type == FLV_STREAM_TYPE_DATA)

        pkt->flags |= AV_PKT_FLAG_KEY;



leave:

    avio_skip(s->pb, 4);

    return ret;

}
