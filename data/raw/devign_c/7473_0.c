static void do_audio_out(AVFormatContext *s, OutputStream *ost,

                         InputStream *ist, AVFrame *decoded_frame)

{

    uint8_t *buftmp;



    int size_out, frame_bytes, resample_changed;

    AVCodecContext *enc = ost->st->codec;

    AVCodecContext *dec = ist->st->codec;

    int osize = av_get_bytes_per_sample(enc->sample_fmt);

    int isize = av_get_bytes_per_sample(dec->sample_fmt);

    uint8_t *buf = decoded_frame->data[0];

    int size     = decoded_frame->nb_samples * dec->channels * isize;



    if (alloc_audio_output_buf(dec, enc, decoded_frame->nb_samples) < 0) {

        av_log(NULL, AV_LOG_FATAL, "Error allocating audio buffer\n");

        exit_program(1);

    }



    if (enc->channels != dec->channels || enc->sample_rate != dec->sample_rate)

        ost->audio_resample = 1;



    resample_changed = ost->resample_sample_fmt  != dec->sample_fmt ||

                       ost->resample_channels    != dec->channels   ||

                       ost->resample_sample_rate != dec->sample_rate;



    if ((ost->audio_resample && !ost->resample) || resample_changed) {

        if (resample_changed) {

            av_log(NULL, AV_LOG_INFO, "Input stream #%d:%d frame changed from rate:%d fmt:%s ch:%d to rate:%d fmt:%s ch:%d\n",

                   ist->file_index, ist->st->index,

                   ost->resample_sample_rate, av_get_sample_fmt_name(ost->resample_sample_fmt), ost->resample_channels,

                   dec->sample_rate, av_get_sample_fmt_name(dec->sample_fmt), dec->channels);

            ost->resample_sample_fmt  = dec->sample_fmt;

            ost->resample_channels    = dec->channels;

            ost->resample_sample_rate = dec->sample_rate;

            if (ost->resample)

                audio_resample_close(ost->resample);

        }

        /* if audio_sync_method is >1 the resampler is needed for audio drift compensation */

        if (audio_sync_method <= 1 &&

            ost->resample_sample_fmt  == enc->sample_fmt &&

            ost->resample_channels    == enc->channels   &&

            ost->resample_sample_rate == enc->sample_rate) {

            ost->resample = NULL;

            ost->audio_resample = 0;

        } else if (ost->audio_resample) {

            if (dec->sample_fmt != AV_SAMPLE_FMT_S16)

                av_log(NULL, AV_LOG_WARNING, "Using s16 intermediate sample format for resampling\n");

            ost->resample = av_audio_resample_init(enc->channels,    dec->channels,

                                                   enc->sample_rate, dec->sample_rate,

                                                   enc->sample_fmt,  dec->sample_fmt,

                                                   16, 10, 0, 0.8);

            if (!ost->resample) {

                av_log(NULL, AV_LOG_FATAL, "Can not resample %d channels @ %d Hz to %d channels @ %d Hz\n",

                       dec->channels, dec->sample_rate,

                       enc->channels, enc->sample_rate);

                exit_program(1);

            }

        }

    }



#define MAKE_SFMT_PAIR(a,b) ((a)+AV_SAMPLE_FMT_NB*(b))

    if (!ost->audio_resample && dec->sample_fmt != enc->sample_fmt &&

        MAKE_SFMT_PAIR(enc->sample_fmt,dec->sample_fmt) != ost->reformat_pair) {

        if (ost->reformat_ctx)

            av_audio_convert_free(ost->reformat_ctx);

        ost->reformat_ctx = av_audio_convert_alloc(enc->sample_fmt, 1,

                                                   dec->sample_fmt, 1, NULL, 0);

        if (!ost->reformat_ctx) {

            av_log(NULL, AV_LOG_FATAL, "Cannot convert %s sample format to %s sample format\n",

                   av_get_sample_fmt_name(dec->sample_fmt),

                   av_get_sample_fmt_name(enc->sample_fmt));

            exit_program(1);

        }

        ost->reformat_pair = MAKE_SFMT_PAIR(enc->sample_fmt,dec->sample_fmt);

    }



    if (audio_sync_method) {

        double delta = get_sync_ipts(ost, ist->last_dts) * enc->sample_rate - ost->sync_opts -

                       av_fifo_size(ost->fifo) / (enc->channels * osize);

        int idelta = delta * dec->sample_rate / enc->sample_rate;

        int byte_delta = idelta * isize * dec->channels;



        // FIXME resample delay

        if (fabs(delta) > 50) {

            if (ist->is_start || fabs(delta) > audio_drift_threshold*enc->sample_rate) {

                if (byte_delta < 0) {

                    byte_delta = FFMAX(byte_delta, -size);

                    size += byte_delta;

                    buf  -= byte_delta;

                    av_log(NULL, AV_LOG_VERBOSE, "discarding %d audio samples\n",

                           -byte_delta / (isize * dec->channels));

                    if (!size)

                        return;

                    ist->is_start = 0;

                } else {

                    av_fast_malloc(&async_buf, &allocated_async_buf_size,

                                   byte_delta + size);

                    if (!async_buf) {

                        av_log(NULL, AV_LOG_FATAL, "Out of memory in do_audio_out\n");

                        exit_program(1);

                    }



                    if (alloc_audio_output_buf(dec, enc, decoded_frame->nb_samples + idelta) < 0) {

                        av_log(NULL, AV_LOG_FATAL, "Error allocating audio buffer\n");

                        exit_program(1);

                    }

                    ist->is_start = 0;



                    generate_silence(async_buf, dec->sample_fmt, byte_delta);

                    memcpy(async_buf + byte_delta, buf, size);

                    buf = async_buf;

                    size += byte_delta;

                    av_log(NULL, AV_LOG_VERBOSE, "adding %d audio samples of silence\n", idelta);

                }

            } else if (audio_sync_method > 1) {

                int comp = av_clip(delta, -audio_sync_method, audio_sync_method);

                av_assert0(ost->audio_resample);

                av_log(NULL, AV_LOG_VERBOSE, "compensating audio timestamp drift:%f compensation:%d in:%d\n",

                       delta, comp, enc->sample_rate);

//                fprintf(stderr, "drift:%f len:%d opts:%"PRId64" ipts:%"PRId64" fifo:%d\n", delta, -1, ost->sync_opts, (int64_t)(get_sync_ipts(ost) * enc->sample_rate), av_fifo_size(ost->fifo)/(ost->st->codec->channels * 2));

                av_resample_compensate(*(struct AVResampleContext**)ost->resample, comp, enc->sample_rate);

            }

        }

    } else

        ost->sync_opts = lrintf(get_sync_ipts(ost, ist->last_dts) * enc->sample_rate) -

                                av_fifo_size(ost->fifo) / (enc->channels * osize); // FIXME wrong



    if (ost->audio_resample) {

        buftmp = audio_buf;

        size_out = audio_resample(ost->resample,

                                  (short *)buftmp, (short *)buf,

                                  size / (dec->channels * isize));

        size_out = size_out * enc->channels * osize;

    } else {

        buftmp = buf;

        size_out = size;

    }



    if (!ost->audio_resample && dec->sample_fmt != enc->sample_fmt) {

        const void *ibuf[6] = { buftmp };

        void *obuf[6]  = { audio_buf };

        int istride[6] = { isize };

        int ostride[6] = { osize };

        int len = size_out / istride[0];

        if (av_audio_convert(ost->reformat_ctx, obuf, ostride, ibuf, istride, len) < 0) {

            printf("av_audio_convert() failed\n");

            if (exit_on_error)

                exit_program(1);

            return;

        }

        buftmp = audio_buf;

        size_out = len * osize;

    }



    /* now encode as many frames as possible */

    if (!(enc->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE)) {

        /* output resampled raw samples */

        if (av_fifo_realloc2(ost->fifo, av_fifo_size(ost->fifo) + size_out) < 0) {

            av_log(NULL, AV_LOG_FATAL, "av_fifo_realloc2() failed\n");

            exit_program(1);

        }

        av_fifo_generic_write(ost->fifo, buftmp, size_out, NULL);



        frame_bytes = enc->frame_size * osize * enc->channels;



        while (av_fifo_size(ost->fifo) >= frame_bytes) {

            av_fifo_generic_read(ost->fifo, audio_buf, frame_bytes, NULL);

            encode_audio_frame(s, ost, audio_buf, frame_bytes);

        }

    } else {

        encode_audio_frame(s, ost, buftmp, size_out);

    }

}
