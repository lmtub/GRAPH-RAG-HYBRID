static int swf_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    SWFContext *swf = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *vst = NULL, *ast = NULL, *st = 0;

    int tag, len, i, frame, v;



    for(;;) {

        uint64_t pos = avio_tell(pb);

        tag = get_swf_tag(pb, &len);

        if (tag < 0)

            return AVERROR(EIO);

        if (tag == TAG_VIDEOSTREAM) {

            int ch_id = avio_rl16(pb);

            len -= 2;



            for (i=0; i<s->nb_streams; i++) {

                st = s->streams[i];

                if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO && st->id == ch_id)

                    goto skip;

            }



            avio_rl16(pb);

            avio_rl16(pb);

            avio_rl16(pb);

            avio_r8(pb);

            /* Check for FLV1 */

            vst = av_new_stream(s, ch_id);

            if (!vst)

                return -1;

            vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            vst->codec->codec_id = ff_codec_get_id(swf_codec_tags, avio_r8(pb));

            av_set_pts_info(vst, 16, 256, swf->frame_rate);

            vst->codec->time_base = (AVRational){ 256, swf->frame_rate };

            len -= 8;

        } else if (tag == TAG_STREAMHEAD || tag == TAG_STREAMHEAD2) {

            /* streaming found */

            int sample_rate_code;



            for (i=0; i<s->nb_streams; i++) {

                st = s->streams[i];

                if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && st->id == -1)

                    goto skip;

            }



            avio_r8(pb);

            v = avio_r8(pb);

            swf->samples_per_frame = avio_rl16(pb);

            ast = av_new_stream(s, -1); /* -1 to avoid clash with video stream ch_id */

            if (!ast)

                return -1;

            ast->codec->channels = 1 + (v&1);

            ast->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            ast->codec->codec_id = ff_codec_get_id(swf_audio_codec_tags, (v>>4) & 15);

            ast->need_parsing = AVSTREAM_PARSE_FULL;

            sample_rate_code= (v>>2) & 3;

            if (!sample_rate_code)

                return AVERROR(EIO);

            ast->codec->sample_rate = 11025 << (sample_rate_code-1);

            av_set_pts_info(ast, 64, 1, ast->codec->sample_rate);

            len -= 4;

        } else if (tag == TAG_VIDEOFRAME) {

            int ch_id = avio_rl16(pb);

            len -= 2;

            for(i=0; i<s->nb_streams; i++) {

                st = s->streams[i];

                if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO && st->id == ch_id) {

                    frame = avio_rl16(pb);

                    av_get_packet(pb, pkt, len-2);

                    pkt->pos = pos;

                    pkt->pts = frame;

                    pkt->stream_index = st->index;

                    return pkt->size;

                }

            }

        } else if (tag == TAG_STREAMBLOCK) {

            for (i = 0; i < s->nb_streams; i++) {

                st = s->streams[i];

                if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO && st->id == -1) {

            if (st->codec->codec_id == CODEC_ID_MP3) {

                avio_skip(pb, 4);

                av_get_packet(pb, pkt, len-4);

            } else { // ADPCM, PCM

                av_get_packet(pb, pkt, len);

            }

            pkt->pos = pos;

            pkt->stream_index = st->index;

            return pkt->size;

                }

            }

        } else if (tag == TAG_JPEG2) {

            for (i=0; i<s->nb_streams; i++) {

                st = s->streams[i];

                if (st->codec->codec_id == CODEC_ID_MJPEG && st->id == -2)

                    break;

            }

            if (i == s->nb_streams) {

                vst = av_new_stream(s, -2); /* -2 to avoid clash with video stream and audio stream */

                if (!vst)

                    return -1;

                vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

                vst->codec->codec_id = CODEC_ID_MJPEG;

                av_set_pts_info(vst, 64, 256, swf->frame_rate);

                vst->codec->time_base = (AVRational){ 256, swf->frame_rate };

                st = vst;

            }

            avio_rl16(pb); /* BITMAP_ID */

            av_new_packet(pkt, len-2);

            avio_read(pb, pkt->data, 4);

            if (AV_RB32(pkt->data) == 0xffd8ffd9 ||

                AV_RB32(pkt->data) == 0xffd9ffd8) {

                /* old SWF files containing SOI/EOI as data start */

                /* files created by swink have reversed tag */

                pkt->size -= 4;

                avio_read(pb, pkt->data, pkt->size);

            } else {

                avio_read(pb, pkt->data + 4, pkt->size - 4);

            }

            pkt->pos = pos;

            pkt->stream_index = st->index;

            return pkt->size;

        }

    skip:

        avio_skip(pb, len);

    }

    return 0;

}
