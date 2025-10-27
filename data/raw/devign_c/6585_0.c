static int ipvideo_decode_block_opcode_0xE(IpvideoContext *s)

{

    int y;

    unsigned char pix;



    /* 1-color encoding: the whole block is 1 solid color */

    CHECK_STREAM_PTR(1);

    pix = *s->stream_ptr++;



    for (y = 0; y < 8; y++) {

        memset(s->pixel_ptr, pix, 8);

        s->pixel_ptr += s->stride;

    }



    /* report success */

    return 0;

}
