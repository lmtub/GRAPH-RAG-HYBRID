static int ea_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    EaDemuxContext *ea = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret = 0;

    int packet_read = 0;

    unsigned int chunk_type, chunk_size;

    int key = 0;

    int av_uninit(num_samples);



    while (!packet_read) {

        chunk_type = avio_rl32(pb);

        chunk_size = (ea->big_endian ? avio_rb32(pb) : avio_rl32(pb)) - 8;



        switch (chunk_type) {

        /* audio data */

        case ISNh_TAG:

            /* header chunk also contains data; skip over the header portion*/

            avio_skip(pb, 32);

            chunk_size -= 32;

        case ISNd_TAG:

        case SCDl_TAG:

        case SNDC_TAG:

        case SDEN_TAG:

            if (!ea->audio_codec) {

                avio_skip(pb, chunk_size);

                break;

            } else if (ea->audio_codec == CODEC_ID_PCM_S16LE_PLANAR ||

                       ea->audio_codec == CODEC_ID_MP3) {

                num_samples = avio_rl32(pb);

                avio_skip(pb, 8);

                chunk_size -= 12;

            }

            ret = av_get_packet(pb, pkt, chunk_size);

            if (ret < 0)

                return ret;

            pkt->stream_index = ea->audio_stream_index;



            switch (ea->audio_codec) {

            case CODEC_ID_ADPCM_EA:

            case CODEC_ID_ADPCM_EA_R1:

            case CODEC_ID_ADPCM_EA_R2:

            case CODEC_ID_ADPCM_IMA_EA_EACS:

                pkt->duration = AV_RL32(pkt->data);

                break;

            case CODEC_ID_ADPCM_EA_R3:

                pkt->duration = AV_RB32(pkt->data);

                break;

            case CODEC_ID_ADPCM_IMA_EA_SEAD:

                pkt->duration = ret * 2 / ea->num_channels;

                break;

            case CODEC_ID_PCM_S16LE_PLANAR:

            case CODEC_ID_MP3:

                pkt->duration = num_samples;

                break;

            default:

                pkt->duration = chunk_size / (ea->bytes * ea->num_channels);

            }



            packet_read = 1;

            break;



        /* ending tag */

        case 0:

        case ISNe_TAG:

        case SCEl_TAG:

        case SEND_TAG:

        case SEEN_TAG:

            ret = AVERROR(EIO);

            packet_read = 1;

            break;



        case MVIh_TAG:

        case kVGT_TAG:

        case pQGT_TAG:

        case TGQs_TAG:

        case MADk_TAG:

            key = AV_PKT_FLAG_KEY;

        case MVIf_TAG:

        case fVGT_TAG:

        case MADm_TAG:

        case MADe_TAG:

            avio_seek(pb, -8, SEEK_CUR);     // include chunk preamble

            chunk_size += 8;

            goto get_video_packet;



        case mTCD_TAG:

            avio_skip(pb, 8);  // skip ea dct header

            chunk_size -= 8;

            goto get_video_packet;



        case MV0K_TAG:

        case MPCh_TAG:

        case pIQT_TAG:

            key = AV_PKT_FLAG_KEY;

        case MV0F_TAG:

get_video_packet:

            ret = av_get_packet(pb, pkt, chunk_size);

            if (ret < 0)

                return ret;

            pkt->stream_index = ea->video_stream_index;

            pkt->flags |= key;

            packet_read = 1;

            break;



        default:

            avio_skip(pb, chunk_size);

            break;

        }

    }



    return ret;

}
