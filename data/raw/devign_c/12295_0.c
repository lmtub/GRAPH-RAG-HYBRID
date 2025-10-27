static int swf_write_audio(AVFormatContext *s, 

                           AVCodecContext *enc, const uint8_t *buf, int size)

{

    SWFContext *swf = s->priv_data;

    int c = 0;



    /* Flash Player limit */

    if ( swf->swf_frame_number >= 16000 ) {

        return 0;

    }



    if (enc->codec_id == CODEC_ID_MP3 ) {

        for (c=0; c<size; c++) {

            swf->audio_fifo[(swf->audio_out_pos+c)%AUDIO_FIFO_SIZE] = buf[c];

        }

        swf->audio_size += size;

        swf->audio_out_pos += size;

        swf->audio_out_pos %= AUDIO_FIFO_SIZE;

    }



    /* if audio only stream make sure we add swf frames */

    if ( swf->video_type == 0 ) {

        swf_write_video(s, enc, 0, 0);

    }



    return 0;

}
