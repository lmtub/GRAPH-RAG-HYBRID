static int flv_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVIOContext *pb      = s->pb;

    AVCodecContext *enc  = s->streams[pkt->stream_index]->codec;

    FLVContext *flv      = s->priv_data;

    FLVStreamContext *sc = s->streams[pkt->stream_index]->priv_data;

    unsigned ts;

    int size = pkt->size;

    uint8_t *data = NULL;

    int flags = 0, flags_size;



    if (enc->codec_id == AV_CODEC_ID_VP6F || enc->codec_id == AV_CODEC_ID_VP6A ||

        enc->codec_id == AV_CODEC_ID_AAC)

        flags_size = 2;

    else if (enc->codec_id == AV_CODEC_ID_H264)

        flags_size = 5;

    else

        flags_size = 1;



    if (flv->delay == AV_NOPTS_VALUE)

        flv->delay = -pkt->dts;



    if (pkt->dts < -flv->delay) {

        av_log(s, AV_LOG_WARNING,

               "Packets are not in the proper order with respect to DTS\n");

        return AVERROR(EINVAL);

    }



    ts = pkt->dts + flv->delay; // add delay to force positive dts



    if (s->event_flags & AVSTREAM_EVENT_FLAG_METADATA_UPDATED) {

        write_metadata(s, ts);

        s->event_flags &= ~AVSTREAM_EVENT_FLAG_METADATA_UPDATED;

    }



    switch (enc->codec_type) {

    case AVMEDIA_TYPE_VIDEO:

        avio_w8(pb, FLV_TAG_TYPE_VIDEO);



        flags = enc->codec_tag;

        if (flags == 0) {

            av_log(s, AV_LOG_ERROR,

                   "video codec %X not compatible with flv\n",

                   enc->codec_id);

            return -1;

        }



        flags |= pkt->flags & AV_PKT_FLAG_KEY ? FLV_FRAME_KEY : FLV_FRAME_INTER;

        break;

    case AVMEDIA_TYPE_AUDIO:

        flags = get_audio_flags(s, enc);



        assert(size);



        avio_w8(pb, FLV_TAG_TYPE_AUDIO);

        break;

    case AVMEDIA_TYPE_DATA:

        avio_w8(pb, FLV_TAG_TYPE_META);

        break;

    default:

        return AVERROR(EINVAL);

    }



    if (enc->codec_id == AV_CODEC_ID_H264)

        /* check if extradata looks like MP4 */

        if (enc->extradata_size > 0 && *(uint8_t*)enc->extradata != 1)

            if (ff_avc_parse_nal_units_buf(pkt->data, &data, &size) < 0)

                return -1;



    /* check Speex packet duration */

    if (enc->codec_id == AV_CODEC_ID_SPEEX && ts - sc->last_ts > 160)

        av_log(s, AV_LOG_WARNING, "Warning: Speex stream has more than "

                                  "8 frames per packet. Adobe Flash "

                                  "Player cannot handle this!\n");



    if (sc->last_ts < ts)

        sc->last_ts = ts;



    avio_wb24(pb, size + flags_size);

    avio_wb24(pb, ts);

    avio_w8(pb, (ts >> 24) & 0x7F); // timestamps are 32 bits _signed_

    avio_wb24(pb, flv->reserved);



    if (enc->codec_type == AVMEDIA_TYPE_DATA) {

        int data_size;

        int64_t metadata_size_pos = avio_tell(pb);

        avio_w8(pb, AMF_DATA_TYPE_STRING);

        put_amf_string(pb, "onTextData");

        avio_w8(pb, AMF_DATA_TYPE_MIXEDARRAY);

        avio_wb32(pb, 2);

        put_amf_string(pb, "type");

        avio_w8(pb, AMF_DATA_TYPE_STRING);

        put_amf_string(pb, "Text");

        put_amf_string(pb, "text");

        avio_w8(pb, AMF_DATA_TYPE_STRING);

        put_amf_string(pb, pkt->data);

        put_amf_string(pb, "");

        avio_w8(pb, AMF_END_OF_OBJECT);

        /* write total size of tag */

        data_size = avio_tell(pb) - metadata_size_pos;

        avio_seek(pb, metadata_size_pos - 10, SEEK_SET);

        avio_wb24(pb, data_size);

        avio_seek(pb, data_size + 10 - 3, SEEK_CUR);

        avio_wb32(pb, data_size + 11);

    } else {

        avio_w8(pb,flags);

        if (enc->codec_id == AV_CODEC_ID_VP6F || enc->codec_id == AV_CODEC_ID_VP6A) {

            if (enc->extradata_size)

                avio_w8(pb, enc->extradata[0]);

            else

                avio_w8(pb, ((FFALIGN(enc->width,  16) - enc->width) << 4) |

                             (FFALIGN(enc->height, 16) - enc->height));

        } else if (enc->codec_id == AV_CODEC_ID_AAC)

            avio_w8(pb, 1); // AAC raw

        else if (enc->codec_id == AV_CODEC_ID_H264) {

            avio_w8(pb, 1); // AVC NALU

            avio_wb24(pb, pkt->pts - pkt->dts);

        }



        avio_write(pb, data ? data : pkt->data, size);



        avio_wb32(pb, size + flags_size + 11); // previous tag size

        flv->duration = FFMAX(flv->duration,

                              pkt->pts + flv->delay + pkt->duration);

    }



    av_free(data);



    return pb->error;

}
