static void mxf_read_pixel_layout(AVIOContext *pb, MXFDescriptor *descriptor)

{

    int code, value, ofs = 0;

    char layout[16] = {0}; /* not for printing, may end up not terminated on purpose */



    do {

        code = avio_r8(pb);

        value = avio_r8(pb);

        av_dlog(NULL, "pixel layout: code %#x\n", code);



        if (ofs <= 14) {

            layout[ofs++] = code;

            layout[ofs++] = value;

        }

    } while (code != 0); /* SMPTE 377M E.2.46 */



    ff_mxf_decode_pixel_layout(layout, &descriptor->pix_fmt);

}
