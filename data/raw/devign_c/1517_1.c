static int avs_read_packet(AVFormatContext * s, AVPacket * pkt)

{

    AvsFormat *avs = s->priv_data;

    int sub_type = 0, size = 0;

    AvsBlockType type = AVS_NONE;

    int palette_size = 0;

    uint8_t palette[4 + 3 * 256];

    int ret;



    if (avs->remaining_audio_size > 0)

        if (avs_read_audio_packet(s, pkt) > 0)

            return 0;



    while (1) {

        if (avs->remaining_frame_size <= 0) {

            if (!avio_rl16(s->pb))    /* found EOF */

                return AVERROR(EIO);

            avs->remaining_frame_size = avio_rl16(s->pb) - 4;

        }



        while (avs->remaining_frame_size > 0) {

            sub_type = avio_r8(s->pb);

            type = avio_r8(s->pb);

            size = avio_rl16(s->pb);

            if (size < 4)


            avs->remaining_frame_size -= size;



            switch (type) {

            case AVS_PALETTE:



                ret = avio_read(s->pb, palette, size - 4);

                if (ret < size - 4)

                    return AVERROR(EIO);

                palette_size = size;

                break;



            case AVS_VIDEO:

                if (!avs->st_video) {

                    avs->st_video = av_new_stream(s, AVS_VIDEO);

                    if (avs->st_video == NULL)

                        return AVERROR(ENOMEM);

                    avs->st_video->codec->codec_type = AVMEDIA_TYPE_VIDEO;

                    avs->st_video->codec->codec_id = CODEC_ID_AVS;

                    avs->st_video->codec->width = avs->width;

                    avs->st_video->codec->height = avs->height;

                    avs->st_video->codec->bits_per_coded_sample=avs->bits_per_sample;

                    avs->st_video->nb_frames = avs->nb_frames;

                    avs->st_video->codec->time_base = (AVRational) {

                    1, avs->fps};

                }

                return avs_read_video_packet(s, pkt, type, sub_type, size,

                                             palette, palette_size);



            case AVS_AUDIO:

                if (!avs->st_audio) {

                    avs->st_audio = av_new_stream(s, AVS_AUDIO);

                    if (avs->st_audio == NULL)

                        return AVERROR(ENOMEM);

                    avs->st_audio->codec->codec_type = AVMEDIA_TYPE_AUDIO;

                }

                avs->remaining_audio_size = size - 4;

                size = avs_read_audio_packet(s, pkt);

                if (size != 0)

                    return size;

                break;



            default:

                avio_skip(s->pb, size - 4);

            }

        }

    }

}