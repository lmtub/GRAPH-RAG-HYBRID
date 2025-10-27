static int swf_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    SWFContext *swf = 0;

    ByteIOContext *pb = &s->pb;

    int nbits, len, frame_rate, tag, v;

    offset_t firstTagOff;

    AVStream *ast = 0;

    AVStream *vst = 0;



    swf = av_malloc(sizeof(SWFContext));

    if (!swf)

        return -1;

    s->priv_data = swf;



    tag = get_be32(pb) & 0xffffff00;



    if (tag == MKBETAG('C', 'W', 'S', 0))

    {

        av_log(s, AV_LOG_ERROR, "Compressed SWF format not supported\n");

        return AVERROR_IO;

    }

    if (tag != MKBETAG('F', 'W', 'S', 0))

        return AVERROR_IO;

    get_le32(pb);

    /* skip rectangle size */

    nbits = get_byte(pb) >> 3;

    len = (4 * nbits - 3 + 7) / 8;

    url_fskip(pb, len);

    frame_rate = get_le16(pb);

    get_le16(pb); /* frame count */



    /* The Flash Player converts 8.8 frame rates

       to milliseconds internally. Do the same to get

       a correct framerate */

    swf->ms_per_frame = ( 1000 * 256 ) / frame_rate;

    swf->samples_per_frame = 0;

    swf->ch_id = -1;



    firstTagOff = url_ftell(pb);

    for(;;) {

        tag = get_swf_tag(pb, &len);

        if (tag < 0) {

            if ( ast || vst ) {

                if ( vst && ast ) {

                    vst->codec->time_base.den = ast->codec->sample_rate / swf->samples_per_frame;

                    vst->codec->time_base.num = 1;

                }

                break;

            }

            av_log(s, AV_LOG_ERROR, "No media found in SWF\n");

            return AVERROR_IO;

        }

        if ( tag == TAG_VIDEOSTREAM && !vst) {

            int codec_id;

            swf->ch_id = get_le16(pb);

            get_le16(pb);

            get_le16(pb);

            get_le16(pb);

            get_byte(pb);

            /* Check for FLV1 */

            codec_id = codec_get_id(swf_codec_tags, get_byte(pb));

            if ( codec_id ) {

                vst = av_new_stream(s, 0);

                av_set_pts_info(vst, 24, 1, 1000); /* 24 bit pts in ms */



                vst->codec->codec_type = CODEC_TYPE_VIDEO;

                vst->codec->codec_id = codec_id;

                if ( swf->samples_per_frame ) {

                    vst->codec->time_base.den = 1000. / swf->ms_per_frame;

                    vst->codec->time_base.num = 1;

                }

            }

        } else if ( ( tag == TAG_STREAMHEAD || tag == TAG_STREAMHEAD2 ) && !ast) {

            /* streaming found */

            get_byte(pb);

            v = get_byte(pb);

            swf->samples_per_frame = get_le16(pb);

            if (len!=4)

                url_fskip(pb,len-4);

            /* if mp3 streaming found, OK */

            if ((v & 0x20) != 0) {

                if ( tag == TAG_STREAMHEAD2 ) {

                    get_le16(pb);

                }

                ast = av_new_stream(s, 1);

                av_set_pts_info(ast, 24, 1, 1000); /* 24 bit pts in ms */

                if (!ast)

                    return -ENOMEM;



                if (v & 0x01)

                    ast->codec->channels = 2;

                else

                    ast->codec->channels = 1;



                switch((v>> 2) & 0x03) {

                case 1:

                    ast->codec->sample_rate = 11025;

                    break;

                case 2:

                    ast->codec->sample_rate = 22050;

                    break;

                case 3:

                    ast->codec->sample_rate = 44100;

                    break;

                default:

                    av_free(ast);

                    return AVERROR_IO;

                }

                ast->codec->codec_type = CODEC_TYPE_AUDIO;

                ast->codec->codec_id = CODEC_ID_MP3;

            }

        } else {

            url_fskip(pb, len);

        }

    }

    url_fseek(pb, firstTagOff, SEEK_SET);



    return 0;

}
