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

    AVCodecContext *enc= ost->st->codec;

    AVCodecContext *dec= ist->st->codec;



    /* SC: dynamic allocation of buffers */

    if (!audio_buf)

        audio_buf = av_malloc(2*MAX_AUDIO_PACKET_SIZE);

    if (!audio_out)

        audio_out = av_malloc(audio_out_size);

    if (!audio_buf || !audio_out)

        return;               /* Should signal an error ! */



    if (enc->channels != dec->channels)

        ost->audio_resample = 1;



    if (ost->audio_resample && !ost->resample) {

        ost->resample = audio_resample_init(enc->channels,    dec->channels,

                                            enc->sample_rate, dec->sample_rate);

        if (!ost->resample) {

            fprintf(stderr, "Can not resample %d channels @ %d Hz to %d channels @ %d Hz\n",

                    dec->channels, dec->sample_rate,

                    enc->channels, enc->sample_rate);

            av_exit(1);

        }

    }



    if(audio_sync_method){

        double delta = get_sync_ipts(ost) * enc->sample_rate - ost->sync_opts

                - av_fifo_size(&ost->fifo)/(ost->st->codec->channels * 2);

        double idelta= delta*ist->st->codec->sample_rate / enc->sample_rate;

        int byte_delta= ((int)idelta)*2*ist->st->codec->channels;



        //FIXME resample delay

        if(fabs(delta) > 50){

            if(ist->is_start || fabs(delta) > audio_drift_threshold*enc->sample_rate){

                if(byte_delta < 0){

                    byte_delta= FFMAX(byte_delta, -size);

                    size += byte_delta;

                    buf  -= byte_delta;

                    if(verbose > 2)

                        fprintf(stderr, "discarding %d audio samples\n", (int)-delta);

                    if(!size)

                        return;

                    ist->is_start=0;

                }else{

                    static uint8_t *input_tmp= NULL;

                    input_tmp= av_realloc(input_tmp, byte_delta + size);



                    if(byte_delta + size <= MAX_AUDIO_PACKET_SIZE)

                        ist->is_start=0;

                    else

                        byte_delta= MAX_AUDIO_PACKET_SIZE - size;



                    memset(input_tmp, 0, byte_delta);

                    memcpy(input_tmp + byte_delta, buf, size);

                    buf= input_tmp;

                    size += byte_delta;

                    if(verbose > 2)

                        fprintf(stderr, "adding %d audio samples of silence\n", (int)delta);

                }

            }else if(audio_sync_method>1){

                int comp= av_clip(delta, -audio_sync_method, audio_sync_method);

                assert(ost->audio_resample);

                if(verbose > 2)

                    fprintf(stderr, "compensating audio timestamp drift:%f compensation:%d in:%d\n", delta, comp, enc->sample_rate);

//                fprintf(stderr, "drift:%f len:%d opts:%"PRId64" ipts:%"PRId64" fifo:%d\n", delta, -1, ost->sync_opts, (int64_t)(get_sync_ipts(ost) * enc->sample_rate), av_fifo_size(&ost->fifo)/(ost->st->codec->channels * 2));

                av_resample_compensate(*(struct AVResampleContext**)ost->resample, comp, enc->sample_rate);

            }

        }

    }else

        ost->sync_opts= lrintf(get_sync_ipts(ost) * enc->sample_rate)

                        - av_fifo_size(&ost->fifo)/(ost->st->codec->channels * 2); //FIXME wrong



    if (ost->audio_resample) {

        buftmp = audio_buf;

        size_out = audio_resample(ost->resample,

                                  (short *)buftmp, (short *)buf,

                                  size / (ist->st->codec->channels * 2));

        size_out = size_out * enc->channels * 2;

    } else {

        buftmp = buf;

        size_out = size;

    }



    /* now encode as many frames as possible */

    if (enc->frame_size > 1) {

        /* output resampled raw samples */

        av_fifo_realloc(&ost->fifo, av_fifo_size(&ost->fifo) + size_out + 1);

        av_fifo_write(&ost->fifo, buftmp, size_out);



        frame_bytes = enc->frame_size * 2 * enc->channels;



        while (av_fifo_read(&ost->fifo, audio_buf, frame_bytes) == 0) {

            AVPacket pkt;

            av_init_packet(&pkt);



            //FIXME pass ost->sync_opts as AVFrame.pts in avcodec_encode_audio()



            ret = avcodec_encode_audio(enc, audio_out, audio_out_size,

                                       (short *)audio_buf);

            audio_size += ret;

            pkt.stream_index= ost->index;

            pkt.data= audio_out;

            pkt.size= ret;

            if(enc->coded_frame && enc->coded_frame->pts != AV_NOPTS_VALUE)

                pkt.pts= av_rescale_q(enc->coded_frame->pts, enc->time_base, ost->st->time_base);

            pkt.flags |= PKT_FLAG_KEY;

            write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);



            ost->sync_opts += enc->frame_size;

        }

    } else {

        AVPacket pkt;

        av_init_packet(&pkt);



        ost->sync_opts += size_out / (2 * enc->channels);



        /* output a pcm frame */

        /* XXX: change encoding codec API to avoid this ? */

        switch(enc->codec->id) {

        case CODEC_ID_PCM_S32LE:

        case CODEC_ID_PCM_S32BE:

        case CODEC_ID_PCM_U32LE:

        case CODEC_ID_PCM_U32BE:

            size_out = size_out << 1;

            break;

        case CODEC_ID_PCM_S24LE:

        case CODEC_ID_PCM_S24BE:

        case CODEC_ID_PCM_U24LE:

        case CODEC_ID_PCM_U24BE:

        case CODEC_ID_PCM_S24DAUD:

            size_out = size_out / 2 * 3;

            break;

        case CODEC_ID_PCM_S16LE:

        case CODEC_ID_PCM_S16BE:

        case CODEC_ID_PCM_U16LE:

        case CODEC_ID_PCM_U16BE:

            break;

        default:

            size_out = size_out >> 1;

            break;

        }

        //FIXME pass ost->sync_opts as AVFrame.pts in avcodec_encode_audio()

        ret = avcodec_encode_audio(enc, audio_out, size_out,

                                   (short *)buftmp);

        audio_size += ret;

        pkt.stream_index= ost->index;

        pkt.data= audio_out;

        pkt.size= ret;

        if(enc->coded_frame && enc->coded_frame->pts != AV_NOPTS_VALUE)

            pkt.pts= av_rescale_q(enc->coded_frame->pts, enc->time_base, ost->st->time_base);

        pkt.flags |= PKT_FLAG_KEY;

        write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);

    }

}
