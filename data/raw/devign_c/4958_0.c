static int ipvideo_decode_block_opcode_0xC(IpvideoContext *s)

{

    int x, y;



    /* 16-color block encoding: each 2x2 block is a different color */

    CHECK_STREAM_PTR(16);



    for (y = 0; y < 8; y += 2) {

        for (x = 0; x < 8; x += 2) {

            s->pixel_ptr[x                ] =

            s->pixel_ptr[x + 1            ] =

            s->pixel_ptr[x +     s->stride] =

            s->pixel_ptr[x + 1 + s->stride] = *s->stream_ptr++;

        }

        s->pixel_ptr += s->stride * 2;

    }



    /* report success */

    return 0;

}
