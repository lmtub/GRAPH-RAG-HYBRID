static int gif_image_write_header(AVIOContext *pb, int width, int height,

                                  int loop_count, uint32_t *palette)

{

    int i;



    avio_write(pb, "GIF", 3);

    avio_write(pb, "89a", 3);

    avio_wl16(pb, width);

    avio_wl16(pb, height);



    if (palette) {

        avio_w8(pb, 0xf7); /* flags: global clut, 256 entries */

        avio_w8(pb, 0x1f); /* background color index */

        avio_w8(pb, 0);    /* aspect ratio */

        for (i = 0; i < 256; i++) {

            const uint32_t v = palette[i] & 0xffffff;

            avio_wb24(pb, v);

        }

    } else {

        avio_w8(pb, 0); /* flags */

        avio_w8(pb, 0); /* background color index */

        avio_w8(pb, 0); /* aspect ratio */

    }



    /* "NETSCAPE EXTENSION" for looped animation GIF */

    avio_w8(pb, 0x21); /* GIF Extension code */

    avio_w8(pb, 0xff); /* Application Extension Label */

    avio_w8(pb, 0x0b); /* Length of Application Block */

    avio_write(pb, "NETSCAPE2.0", sizeof("NETSCAPE2.0") - 1);

    avio_w8(pb, 0x03); /* Length of Data Sub-Block */

    avio_w8(pb, 0x01);

    avio_wl16(pb, (uint16_t)loop_count);

    avio_w8(pb, 0x00); /* Data Sub-block Terminator */



    return 0;

}
