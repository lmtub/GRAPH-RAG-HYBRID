static int mov_get_codec_tag(AVFormatContext *s, MOVTrack *track)

{

    int tag = track->enc->codec_tag;



    if (!tag || (track->enc->strict_std_compliance >= FF_COMPLIANCE_NORMAL &&

                 (tag == MKTAG('d','v','c','p') ||

                  track->enc->codec_id == CODEC_ID_RAWVIDEO ||

                  track->enc->codec_id == CODEC_ID_H263 ||

                  av_get_bits_per_sample(track->enc->codec_id)))) { // pcm audio

        if (track->enc->codec_id == CODEC_ID_DVVIDEO)

            tag = mov_get_dv_codec_tag(s, track);

        else if (track->enc->codec_id == CODEC_ID_RAWVIDEO)

            tag = mov_get_rawvideo_codec_tag(s, track);

        else if (track->enc->codec_type == AVMEDIA_TYPE_VIDEO) {

            tag = ff_codec_get_tag(codec_movvideo_tags, track->enc->codec_id);

            if (!tag) { // if no mac fcc found, try with Microsoft tags

                tag = ff_codec_get_tag(ff_codec_bmp_tags, track->enc->codec_id);

                if (tag)

                    av_log(s, AV_LOG_INFO, "Warning, using MS style video codec tag, "

                           "the file may be unplayable!\n");

            }

        } else if (track->enc->codec_type == AVMEDIA_TYPE_AUDIO) {

            tag = ff_codec_get_tag(codec_movaudio_tags, track->enc->codec_id);

            if (!tag) { // if no mac fcc found, try with Microsoft tags

                int ms_tag = ff_codec_get_tag(ff_codec_wav_tags, track->enc->codec_id);

                if (ms_tag) {

                    tag = MKTAG('m', 's', ((ms_tag >> 8) & 0xff), (ms_tag & 0xff));

                    av_log(s, AV_LOG_INFO, "Warning, using MS style audio codec tag, "

                           "the file may be unplayable!\n");

                }

            }

        } else if (track->enc->codec_type == AVMEDIA_TYPE_SUBTITLE)

            tag = ff_codec_get_tag(ff_codec_movsubtitle_tags, track->enc->codec_id);

    }



    return tag;

}
