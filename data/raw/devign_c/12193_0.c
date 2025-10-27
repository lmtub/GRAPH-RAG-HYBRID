static int has_decode_delay_been_guessed(AVStream *st)

{

    return st->codec->codec_id != CODEC_ID_H264 ||

        st->codec_info_nb_frames >= 6 + st->codec->has_b_frames;

}
