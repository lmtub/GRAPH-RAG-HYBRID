static int ipvideo_decode_block_opcode_0xA(IpvideoContext *s)

{

    int x, y;

    unsigned char P[4];

    int flags = 0;



    /* 4-color encoding for each 4x4 quadrant, or 4-color encoding on

     * either top and bottom or left and right halves */

    CHECK_STREAM_PTR(24);



    if (s->stream_ptr[0] <= s->stream_ptr[1]) {



        /* 4-color encoding for each quadrant; need 32 bytes */

        CHECK_STREAM_PTR(32);



        for (y = 0; y < 16; y++) {

            // new values for each 4x4 block

            if (!(y & 3)) {

                memcpy(P, s->stream_ptr, 4);

                s->stream_ptr += 4;

                flags = bytestream_get_le32(&s->stream_ptr);

            }



            for (x = 0; x < 4; x++, flags >>= 2)

                *s->pixel_ptr++ = P[flags & 0x03];



            s->pixel_ptr += s->stride - 4;

            // switch to right half

            if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;

        }



    } else {

        // vertical split?

        int vert = s->stream_ptr[12] <= s->stream_ptr[13];

        uint64_t flags = 0;



        /* 4-color encoding for either left and right or top and bottom

         * halves */



        for (y = 0; y < 16; y++) {

            // load values for each half

            if (!(y & 7)) {

                memcpy(P, s->stream_ptr, 4);

                s->stream_ptr += 4;

                flags = bytestream_get_le64(&s->stream_ptr);

            }



            for (x = 0; x < 4; x++, flags >>= 2)

                *s->pixel_ptr++ = P[flags & 0x03];



            if (vert) {

                s->pixel_ptr += s->stride - 4;

                // switch to right half

                if (y == 7) s->pixel_ptr -= 8 * s->stride - 4;

            } else if (y & 1) s->pixel_ptr += s->line_inc;

        }

    }



    /* report success */

    return 0;

}
