static int flv_write_packet(AVFormatContext *s, int stream_index,

                            const uint8_t *buf, int size, int64_t timestamp)

{

    ByteIOContext *pb = &s->pb;

    AVCodecContext *enc = &s->streams[stream_index]->codec;

    FLVContext *flv = s->priv_data;



    if (enc->codec_type == CODEC_TYPE_VIDEO) {

        FLVFrame *frame = av_malloc(sizeof(FLVFrame));

        frame->next = 0;

        frame->type = 9;

        frame->flags = 2; // choose h263

        frame->flags |= enc->coded_frame->key_frame ? 0x10 : 0x20; // add keyframe indicator

        frame->timestamp = timestamp;

        //frame->timestamp = ( ( flv->frameCount * (int64_t)FRAME_RATE_BASE * (int64_t)1000 ) / (int64_t)enc->frame_rate );

        //printf("%08x %f %f\n",frame->timestamp,(double)enc->frame_rate/(double)FRAME_RATE_BASE,1000*(double)FRAME_RATE_BASE/(double)enc->frame_rate);

        frame->size = size;

        frame->data = av_malloc(size);

        memcpy(frame->data,buf,size);

        flv->hasVideo = 1;



        InsertSorted(flv,frame);



        flv->frameCount ++;

    }

    else if (enc->codec_type == CODEC_TYPE_AUDIO) {

#ifdef CONFIG_MP3LAME

        if (enc->codec_id == CODEC_ID_MP3 ) {

            int c=0;

            for (;c<size;c++) {

                flv->audioFifo[(flv->audioOutPos+c)%AUDIO_FIFO_SIZE] = buf[c];

            }

            flv->audioSize += size;

            flv->audioOutPos += size;

            flv->audioOutPos %= AUDIO_FIFO_SIZE;



            if ( flv->initDelay == -1 ) {

                flv->initDelay = timestamp;

            }



            if ( flv->audioTime == -1 ) {

                flv->audioTime = timestamp;

//                flv->audioTime = ( ( ( flv->sampleCount - enc->delay ) * 8000 ) / flv->audioRate ) - flv->initDelay - 250;

//                if ( flv->audioTime < 0 ) {

//                    flv->audioTime = 0;

//                }

            }

        }

        for ( ; flv->audioSize >= 4 ; ) {



            int mp3FrameSize = 0;

            int mp3SampleRate = 0;

            int mp3IsMono = 0;

            int mp3SamplesPerFrame = 0;



            if ( mp3info(&flv->audioFifo[flv->audioInPos],&mp3FrameSize,&mp3SamplesPerFrame,&mp3SampleRate,&mp3IsMono) ) {

                if ( flv->audioSize >= mp3FrameSize ) {



                    int soundFormat = 0x22;

                    int c=0;

                    FLVFrame *frame = av_malloc(sizeof(FLVFrame));



                    flv->audioRate = mp3SampleRate;



                    switch (mp3SampleRate) {

                        case    44100:

                            soundFormat |= 0x0C;

                            break;

                        case    22050:

                            soundFormat |= 0x08;

                            break;

                        case    11025:

                            soundFormat |= 0x04;

                            break;

                    }



                    if ( !mp3IsMono ) {

                        soundFormat |= 0x01;

                    }



                    frame->next = 0;

                    frame->type = 8;

                    frame->flags = soundFormat;

                    frame->timestamp = flv->audioTime;

                    frame->size = mp3FrameSize;

                    frame->data = av_malloc(mp3FrameSize);



                    for (;c<mp3FrameSize;c++) {

                        frame->data[c] = flv->audioFifo[(flv->audioInPos+c)%AUDIO_FIFO_SIZE];

                    }



                    flv->audioInPos += mp3FrameSize;

                    flv->audioSize -= mp3FrameSize;

                    flv->audioInPos %= AUDIO_FIFO_SIZE;

                    flv->sampleCount += mp3SamplesPerFrame;



                    // Reset audio for next round

                    flv->audioTime = -1;

                    // We got audio! Make sure we set this to the global flags on closure

                    flv->hasAudio = 1;



                    InsertSorted(flv,frame);

                }

                break;

            }

            flv->audioInPos ++;

            flv->audioSize --;

            flv->audioInPos %= AUDIO_FIFO_SIZE;

            // no audio in here!

            flv->audioTime = -1;

        }

#endif

    }

    Dump(flv,pb,128);

    put_flush_packet(pb);

    return 0;

}
