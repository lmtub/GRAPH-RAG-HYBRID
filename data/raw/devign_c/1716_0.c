static int rm_write_video(AVFormatContext *s, const uint8_t *buf, int size)

{

    RMContext *rm = s->priv_data;

    ByteIOContext *pb = &s->pb;

    StreamInfo *stream = rm->video_stream;

    int key_frame = stream->enc->coded_frame->key_frame;



    /* XXX: this is incorrect: should be a parameter */



    /* Well, I spent some time finding the meaning of these bits. I am

       not sure I understood everything, but it works !! */

#if 1

    write_packet_header(s, stream, size + 7, key_frame);

    /* bit 7: '1' if final packet of a frame converted in several packets */

    put_byte(pb, 0x81); 

    /* bit 7: '1' if I frame. bits 6..0 : sequence number in current

       frame starting from 1 */

    if (key_frame) {

        put_byte(pb, 0x81); 

    } else {

        put_byte(pb, 0x01); 

    }

    put_be16(pb, 0x4000 | (size)); /* total frame size */

    put_be16(pb, 0x4000 | (size));              /* offset from the start or the end */

#else

    /* full frame */

    write_packet_header(s, size + 6);

    put_byte(pb, 0xc0); 

    put_be16(pb, 0x4000 | size); /* total frame size */

    put_be16(pb, 0x4000 + packet_number * 126); /* position in stream */

#endif

    put_byte(pb, stream->nb_frames & 0xff); 

    

    put_buffer(pb, buf, size);

    put_flush_packet(pb);



    stream->nb_frames++;

    return 0;

}
