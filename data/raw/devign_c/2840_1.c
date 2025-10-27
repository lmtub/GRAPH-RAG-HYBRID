static void build_basic_mjpeg_vlc(MJpegDecodeContext *s)

{

    build_vlc(&s->vlcs[0][0], avpriv_mjpeg_bits_dc_luminance,

              avpriv_mjpeg_val_dc, 12, 0, 0);

    build_vlc(&s->vlcs[0][1], avpriv_mjpeg_bits_dc_chrominance,

              avpriv_mjpeg_val_dc, 12, 0, 0);

    build_vlc(&s->vlcs[1][0], avpriv_mjpeg_bits_ac_luminance,

              avpriv_mjpeg_val_ac_luminance, 251, 0, 1);

    build_vlc(&s->vlcs[1][1], avpriv_mjpeg_bits_ac_chrominance,

              avpriv_mjpeg_val_ac_chrominance, 251, 0, 1);

    build_vlc(&s->vlcs[2][0], avpriv_mjpeg_bits_ac_luminance,

              avpriv_mjpeg_val_ac_luminance, 251, 0, 0);

    build_vlc(&s->vlcs[2][1], avpriv_mjpeg_bits_ac_chrominance,

              avpriv_mjpeg_val_ac_chrominance, 251, 0, 0);

}
