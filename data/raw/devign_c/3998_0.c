static int rm_write_audio(AVFormatContext *s, const uint8_t *buf, int size)

{

    uint8_t *buf1;

    RMContext *rm = s->priv_data;

    ByteIOContext *pb = &s->pb;

    StreamInfo *stream = rm->audio_stream;

    int i;



    /* XXX: suppress this malloc */

    buf1= (uint8_t*) av_malloc( size * sizeof(uint8_t) );

    

    write_packet_header(s, stream, size, stream->enc->coded_frame->key_frame);

    

    /* for AC3, the words seems to be reversed */

    for(i=0;i<size;i+=2) {

        buf1[i] = buf[i+1];

        buf1[i+1] = buf[i];

    }

    put_buffer(pb, buf1, size);

    put_flush_packet(pb);

    stream->nb_frames++;

    av_free(buf1);

    return 0;

}
