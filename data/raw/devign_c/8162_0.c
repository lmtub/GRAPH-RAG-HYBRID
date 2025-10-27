static int ipvideo_decode_block_opcode_0xB(IpvideoContext *s)

{

    int y;



    /* 64-color encoding (each pixel in block is a different color) */

    CHECK_STREAM_PTR(64);



    for (y = 0; y < 8; y++) {

        memcpy(s->pixel_ptr, s->stream_ptr, 8);

        s->stream_ptr += 8;

        s->pixel_ptr  += s->stride;

    }



    /* report success */

    return 0;

}
