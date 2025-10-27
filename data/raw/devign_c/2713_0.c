static int send_mono_rect(VncState *vs, int w, int h, uint32_t bg, uint32_t fg)

{

    size_t bytes;

    int stream = 1;

    int level = tight_conf[vs->tight_compression].mono_zlib_level;



    bytes = ((w + 7) / 8) * h;



    vnc_write_u8(vs, (stream | VNC_TIGHT_EXPLICIT_FILTER) << 4);

    vnc_write_u8(vs, VNC_TIGHT_FILTER_PALETTE);

    vnc_write_u8(vs, 1);



    switch(vs->clientds.pf.bytes_per_pixel) {

    case 4:

    {

        uint32_t buf[2] = {bg, fg};

        size_t ret = sizeof (buf);



        if (vs->tight_pixel24) {

            tight_pack24(vs, (unsigned char*)buf, 2, &ret);

        }

        vnc_write(vs, buf, ret);



        tight_encode_mono_rect32(vs->tight.buffer, w, h, bg, fg);

        break;

    }

    case 2:

        vnc_write(vs, &bg, 2);

        vnc_write(vs, &fg, 2);

        tight_encode_mono_rect16(vs->tight.buffer, w, h, bg, fg);

        break;

    default:

        vnc_write_u8(vs, bg);

        vnc_write_u8(vs, fg);

        tight_encode_mono_rect8(vs->tight.buffer, w, h, bg, fg);

        break;

    }

    vs->tight.offset = bytes;



    bytes = tight_compress_data(vs, stream, bytes, level, Z_DEFAULT_STRATEGY);

    return (bytes >= 0);

}
