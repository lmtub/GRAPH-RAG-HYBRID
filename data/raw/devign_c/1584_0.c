static int gif_image_write_header(uint8_t **bytestream,

                                  int width, int height, int loop_count,

                                  uint32_t *palette)

{

    int i;

    unsigned int v;



    bytestream_put_buffer(bytestream, "GIF", 3);

    bytestream_put_buffer(bytestream, "89a", 3);

    bytestream_put_le16(bytestream, width);

    bytestream_put_le16(bytestream, height);



    bytestream_put_byte(bytestream, 0xf7); /* flags: global clut, 256 entries */

    bytestream_put_byte(bytestream, 0x1f); /* background color index */

    bytestream_put_byte(bytestream, 0); /* aspect ratio */



    /* the global palette */

    if (!palette) {

        bytestream_put_buffer(bytestream, (const unsigned char *)gif_clut, 216*3);

        for(i=0;i<((256-216)*3);i++)

            bytestream_put_byte(bytestream, 0);

    } else {

        for(i=0;i<256;i++) {

            v = palette[i];

            bytestream_put_be24(bytestream, v);

        }

    }



        /*        update: this is the 'NETSCAPE EXTENSION' that allows for looped animated gif

                see http://members.aol.com/royalef/gifabout.htm#net-extension



                byte   1       : 33 (hex 0x21) GIF Extension code

                byte   2       : 255 (hex 0xFF) Application Extension Label

                byte   3       : 11 (hex (0x0B) Length of Application Block

                                         (eleven bytes of data to follow)

                bytes  4 to 11 : "NETSCAPE"

                bytes 12 to 14 : "2.0"

                byte  15       : 3 (hex 0x03) Length of Data Sub-Block

                                         (three bytes of data to follow)

                byte  16       : 1 (hex 0x01)

                bytes 17 to 18 : 0 to 65535, an unsigned integer in

                                         lo-hi byte format. This indicate the

                                         number of iterations the loop should

                                         be executed.

                bytes 19       : 0 (hex 0x00) a Data Sub-block Terminator

        */



    /* application extension header */

#ifdef GIF_ADD_APP_HEADER

    if (loop_count >= 0 && loop_count <= 65535) {

        bytestream_put_byte(bytestream, 0x21);

        bytestream_put_byte(bytestream, 0xff);

        bytestream_put_byte(bytestream, 0x0b);

        bytestream_put_buffer(bytestream, "NETSCAPE2.0", 11);  // bytes 4 to 14

        bytestream_put_byte(bytestream, 0x03); // byte 15

        bytestream_put_byte(bytestream, 0x01); // byte 16

        bytestream_put_le16(bytestream, (uint16_t)loop_count);

        bytestream_put_byte(bytestream, 0x00); // byte 19

    }

#endif

    return 0;

}
