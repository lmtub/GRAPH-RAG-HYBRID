static av_cold int jpg_init(AVCodecContext *avctx, JPGContext *c)

{

    int ret;



    ret = build_vlc(&c->dc_vlc[0], avpriv_mjpeg_bits_dc_luminance,

                    avpriv_mjpeg_val_dc, 12, 0);

    if (ret)

        return ret;

    ret = build_vlc(&c->dc_vlc[1], avpriv_mjpeg_bits_dc_chrominance,

                    avpriv_mjpeg_val_dc, 12, 0);

    if (ret)

        return ret;

    ret = build_vlc(&c->ac_vlc[0], avpriv_mjpeg_bits_ac_luminance,

                    avpriv_mjpeg_val_ac_luminance, 251, 1);

    if (ret)

        return ret;

    ret = build_vlc(&c->ac_vlc[1], avpriv_mjpeg_bits_ac_chrominance,

                    avpriv_mjpeg_val_ac_chrominance, 251, 1);

    if (ret)

        return ret;



    ff_blockdsp_init(&c->bdsp, avctx);

    ff_idctdsp_init(&c->idsp, avctx);

    ff_init_scantable(c->idsp.idct_permutation, &c->scantable,

                      ff_zigzag_direct);



    return 0;

}
