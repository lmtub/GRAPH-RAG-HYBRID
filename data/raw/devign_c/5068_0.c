static int fourxm_read_packet(AVFormatContext *s,

                              AVPacket *pkt)

{

    FourxmDemuxContext *fourxm = s->priv_data;

    ByteIOContext *pb = s->pb;

    unsigned int fourcc_tag;

    unsigned int size, out_size;

    int ret = 0;

    unsigned int track_number;

    int packet_read = 0;

    unsigned char header[8];

    int audio_frame_count;



    while (!packet_read) {



        if ((ret = get_buffer(s->pb, header, 8)) < 0)

            return ret;

        fourcc_tag = AV_RL32(&header[0]);

        size = AV_RL32(&header[4]);

        if (url_feof(pb))

            return AVERROR(EIO);

        switch (fourcc_tag) {



        case LIST_TAG:

            /* this is a good time to bump the video pts */

            fourxm->video_pts ++;



            /* skip the LIST-* tag and move on to the next fourcc */

            get_le32(pb);

            break;



        case ifrm_TAG:

        case pfrm_TAG:

        case cfrm_TAG:

        case ifr2_TAG:

        case pfr2_TAG:

        case cfr2_TAG:

            /* allocate 8 more bytes than 'size' to account for fourcc

             * and size */

            if (size + 8 < size || av_new_packet(pkt, size + 8))

                return AVERROR(EIO);

            pkt->stream_index = fourxm->video_stream_index;

            pkt->pts = fourxm->video_pts;

            pkt->pos = url_ftell(s->pb);

            memcpy(pkt->data, header, 8);

            ret = get_buffer(s->pb, &pkt->data[8], size);



            if (ret < 0){

                av_free_packet(pkt);

            }else

                packet_read = 1;

            break;



        case snd__TAG:

            track_number = get_le32(pb);

            out_size= get_le32(pb);

            size-=8;



            if (track_number < fourxm->track_count) {

                ret= av_get_packet(s->pb, pkt, size);

                if(ret<0)

                    return AVERROR(EIO);

                pkt->stream_index =

                    fourxm->tracks[track_number].stream_index;

                pkt->pts = fourxm->tracks[track_number].audio_pts;

                packet_read = 1;



                /* pts accounting */

                audio_frame_count = size;

                if (fourxm->tracks[track_number].adpcm)

                    audio_frame_count -=

                        2 * (fourxm->tracks[track_number].channels);

                audio_frame_count /=

                      fourxm->tracks[track_number].channels;

                if (fourxm->tracks[track_number].adpcm){

                    audio_frame_count *= 2;

                }else

                    audio_frame_count /=

                    (fourxm->tracks[track_number].bits / 8);

                fourxm->tracks[track_number].audio_pts += audio_frame_count;



            } else {

                url_fseek(pb, size, SEEK_CUR);

            }

            break;



        default:

            url_fseek(pb, size, SEEK_CUR);

            break;

        }

    }

    return ret;

}
