static int swf_write_video(AVFormatContext *s, 

                           AVCodecContext *enc, const uint8_t *buf, int size)

{

    ByteIOContext *pb = &s->pb;

    static int tag_id = 0;



    if (enc->frame_number > 1) {

        /* remove the shape */

        put_swf_tag(s, TAG_REMOVEOBJECT);

        put_le16(pb, SHAPE_ID); /* shape ID */

        put_le16(pb, 1); /* depth */

        put_swf_end_tag(s);

        

        /* free the bitmap */

        put_swf_tag(s, TAG_FREECHARACTER);

        put_le16(pb, BITMAP_ID);

        put_swf_end_tag(s);

    }



    put_swf_tag(s, TAG_JPEG2 | TAG_LONG);



    put_le16(pb, tag_id); /* ID of the image */



    /* a dummy jpeg header seems to be required */

    put_byte(pb, 0xff); 

    put_byte(pb, 0xd8);

    put_byte(pb, 0xff);

    put_byte(pb, 0xd9);

    /* write the jpeg image */

    put_buffer(pb, buf, size);



    put_swf_end_tag(s);



    /* draw the shape */



    put_swf_tag(s, TAG_PLACEOBJECT);

    put_le16(pb, SHAPE_ID); /* shape ID */

    put_le16(pb, 1); /* depth */

    put_swf_matrix(pb, 1 << FRAC_BITS, 0, 0, 1 << FRAC_BITS, 0, 0);

    put_swf_end_tag(s);

    

    /* output the frame */

    put_swf_tag(s, TAG_SHOWFRAME);

    put_swf_end_tag(s);

    

    put_flush_packet(&s->pb);

    return 0;

}
