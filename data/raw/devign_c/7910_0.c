static int flv_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVIOContext *pb = s->pb;

    AVCodecContext *enc = s->streams[pkt->stream_index]->codec;

    FLVContext *flv = s->priv_data;

    unsigned ts;

    int size= pkt->size;

    uint8_t *data= NULL;

    int flags, flags_size;



//    av_log(s, AV_LOG_DEBUG, "type:%d pts: %"PRId64" size:%d\n", enc->codec_type, timestamp, size);



    if(enc->codec_id == CODEC_ID_VP6 || enc->codec_id == CODEC_ID_VP6F ||

       enc->codec_id == CODEC_ID_AAC)

        flags_size= 2;

    else if(enc->codec_id == CODEC_ID_H264)

        flags_size= 5;

    else

        flags_size= 1;



    if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {

        avio_w8(pb, FLV_TAG_TYPE_VIDEO);



        flags = enc->codec_tag;

        if(flags == 0) {

            av_log(enc, AV_LOG_ERROR, "video codec %X not compatible with flv\n",enc->codec_id);

            return -1;

        }



        flags |= pkt->flags & AV_PKT_FLAG_KEY ? FLV_FRAME_KEY : FLV_FRAME_INTER;

    } else {

        assert(enc->codec_type == AVMEDIA_TYPE_AUDIO);

        flags = get_audio_flags(enc);



        assert(size);



        avio_w8(pb, FLV_TAG_TYPE_AUDIO);

    }



    if (enc->codec_id == CODEC_ID_H264) {

        /* check if extradata looks like mp4 formated */

        if (enc->extradata_size > 0 && *(uint8_t*)enc->extradata != 1) {

            if (ff_avc_parse_nal_units_buf(pkt->data, &data, &size) < 0)

                return -1;

        }

    }

    if (!flv->delay && pkt->dts < 0)

        flv->delay = -pkt->dts;



    ts = pkt->dts + flv->delay; // add delay to force positive dts

    if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (flv->last_video_ts < ts)

            flv->last_video_ts = ts;

    }

    avio_wb24(pb,size + flags_size);

    avio_wb24(pb,ts);

    avio_w8(pb,(ts >> 24) & 0x7F); // timestamps are 32bits _signed_

    avio_wb24(pb,flv->reserved);

    avio_w8(pb,flags);

    if (enc->codec_id == CODEC_ID_VP6)

        avio_w8(pb,0);

    if (enc->codec_id == CODEC_ID_VP6F)

        avio_w8(pb, enc->extradata_size ? enc->extradata[0] : 0);

    else if (enc->codec_id == CODEC_ID_AAC)

        avio_w8(pb,1); // AAC raw

    else if (enc->codec_id == CODEC_ID_H264) {

        avio_w8(pb,1); // AVC NALU

        avio_wb24(pb,pkt->pts - pkt->dts);

    }



    avio_write(pb, data ? data : pkt->data, size);



    avio_wb32(pb,size+flags_size+11); // previous tag size

    flv->duration = FFMAX(flv->duration, pkt->pts + flv->delay + pkt->duration);



    avio_flush(pb);



    av_free(data);



    return pb->error;

}
