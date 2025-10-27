static void do_audio_out(AVFormatContext *s, 

                         AVOutputStream *ost, 

                         AVInputStream *ist,

                         unsigned char *buf, int size)

{

    uint8_t *buftmp;

    static uint8_t *audio_buf = NULL;

    static uint8_t *audio_out = NULL;

    const int audio_out_size= 4*MAX_AUDIO_PACKET_SIZE;



    int size_out, frame_bytes, ret;

    AVCodecContext *enc;



    /* SC: dynamic allocation of buffers */

    if (!audio_buf)

        audio_buf = av_malloc(2*MAX_AUDIO_PACKET_SIZE);

    if (!audio_out)

        audio_out = av_malloc(audio_out_size);

    if (!audio_buf || !audio_out)

        return;               /* Should signal an error ! */



    

    enc = &ost->st->codec;



    if (ost->audio_resample) {

        buftmp = audio_buf;

        size_out = audio_resample(ost->resample, 

                                  (short *)buftmp, (short *)buf,

                                  size / (ist->st->codec.channels * 2));

        size_out = size_out * enc->channels * 2;

    } else {

        buftmp = buf;

        size_out = size;

    }



    /* now encode as many frames as possible */

    if (enc->frame_size > 1) {

        /* output resampled raw samples */

        fifo_write(&ost->fifo, buftmp, size_out, 

                   &ost->fifo.wptr);



        frame_bytes = enc->frame_size * 2 * enc->channels;

        

        while (fifo_read(&ost->fifo, audio_buf, frame_bytes, 

                     &ost->fifo.rptr) == 0) {

            AVPacket pkt;

            av_init_packet(&pkt);



            ret = avcodec_encode_audio(enc, audio_out, audio_out_size, 

                                       (short *)audio_buf);

            audio_size += ret;

            pkt.stream_index= ost->index;

            pkt.data= audio_out;

            pkt.size= ret;

            if(enc->coded_frame)

                pkt.pts= enc->coded_frame->pts;

            pkt.flags |= PKT_FLAG_KEY;

            av_write_frame(s, &pkt);

        }

    } else {

        AVPacket pkt;

        av_init_packet(&pkt);

        /* output a pcm frame */

        /* XXX: change encoding codec API to avoid this ? */

        switch(enc->codec->id) {

        case CODEC_ID_PCM_S16LE:

        case CODEC_ID_PCM_S16BE:

        case CODEC_ID_PCM_U16LE:

        case CODEC_ID_PCM_U16BE:

            break;

        default:

            size_out = size_out >> 1;

            break;

        }

        ret = avcodec_encode_audio(enc, audio_out, size_out, 

				   (short *)buftmp);

        audio_size += ret;

        pkt.stream_index= ost->index;

        pkt.data= audio_out;

        pkt.size= ret;

        if(enc->coded_frame)

            pkt.pts= enc->coded_frame->pts;

        pkt.flags |= PKT_FLAG_KEY;

        av_write_frame(s, &pkt);

    }

}
