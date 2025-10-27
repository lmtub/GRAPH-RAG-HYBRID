static int flv_set_video_codec(AVFormatContext *s, AVStream *vstream, int flv_codecid) {

    AVCodecContext *vcodec = vstream->codec;

    switch(flv_codecid) {

        case FLV_CODECID_H263  : vcodec->codec_id = CODEC_ID_FLV1   ; break;

        case FLV_CODECID_REALH263: vcodec->codec_id = CODEC_ID_H263 ; break; // Really mean it this time

        case FLV_CODECID_SCREEN: vcodec->codec_id = CODEC_ID_FLASHSV; break;

        case FLV_CODECID_SCREEN2: vcodec->codec_id = CODEC_ID_FLASHSV2; break;

        case FLV_CODECID_VP6   : vcodec->codec_id = CODEC_ID_VP6F   ;

        case FLV_CODECID_VP6A  :

            if(flv_codecid == FLV_CODECID_VP6A)

                vcodec->codec_id = CODEC_ID_VP6A;

            if(vcodec->extradata_size != 1) {

                vcodec->extradata_size = 1;

                vcodec->extradata = av_malloc(1);

            }

            vcodec->extradata[0] = avio_r8(s->pb);

            return 1; // 1 byte body size adjustment for flv_read_packet()

        case FLV_CODECID_H264:

            vcodec->codec_id = CODEC_ID_H264;

            return 3; // not 4, reading packet type will consume one byte

        case FLV_CODECID_MPEG4:

            vcodec->codec_id = CODEC_ID_MPEG4;

            return 3;

        default:

            av_log(s, AV_LOG_INFO, "Unsupported video codec (%x)\n", flv_codecid);

            vcodec->codec_tag = flv_codecid;

    }



    return 0;

}
