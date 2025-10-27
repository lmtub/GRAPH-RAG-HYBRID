static void init_band_stepsize(AVCodecContext *avctx,

                               Jpeg2000Band *band,

                               Jpeg2000CodingStyle *codsty,

                               Jpeg2000QuantStyle *qntsty,

                               int bandno, int gbandno, int reslevelno,

                               int cbps)

{

    /* TODO: Implementation of quantization step not finished,

     * see ISO/IEC 15444-1:2002 E.1 and A.6.4. */

    switch (qntsty->quantsty) {

        uint8_t gain;

    case JPEG2000_QSTY_NONE:

        /* TODO: to verify. No quantization in this case */

        band->f_stepsize = 1;

        break;

    case JPEG2000_QSTY_SI:

        /*TODO: Compute formula to implement. */

//         numbps = cbps +

//                  lut_gain[codsty->transform == FF_DWT53][bandno + (reslevelno > 0)];

//         band->f_stepsize = SHL(2048 + qntsty->mant[gbandno],

//                                2 + numbps - qntsty->expn[gbandno]);

//         break;

    case JPEG2000_QSTY_SE:

        /* Exponent quantization step.

         * Formula:

         * delta_b = 2 ^ (R_b - expn_b) * (1 + (mant_b / 2 ^ 11))

         * R_b = R_I + log2 (gain_b )

         * see ISO/IEC 15444-1:2002 E.1.1 eqn. E-3 and E-4 */

        gain            = cbps;

        band->f_stepsize  = pow(2.0, gain - qntsty->expn[gbandno]);

        band->f_stepsize *= qntsty->mant[gbandno] / 2048.0 + 1.0;

        break;

    default:

        band->f_stepsize = 0;

        av_log(avctx, AV_LOG_ERROR, "Unknown quantization format\n");

        break;

    }

    if (codsty->transform != FF_DWT53) {

        int lband = 0;

        switch (bandno + (reslevelno > 0)) {

            case 1:

            case 2:

                band->f_stepsize *= F_LFTG_X * 2;

                lband = 1;

                break;

            case 3:

                band->f_stepsize *= F_LFTG_X * F_LFTG_X * 4;

                break;

        }

        if (codsty->transform == FF_DWT97) {

            band->f_stepsize *= pow(F_LFTG_K, 2*(codsty->nreslevels2decode - reslevelno) + lband - 2);

        }

    }



    band->i_stepsize = band->f_stepsize * (1 << 15);



    /* FIXME: In openjepg code stespize = stepsize * 0.5. Why?

     * If not set output of entropic decoder is not correct. */

    if (!av_codec_is_encoder(avctx->codec))

        band->f_stepsize *= 0.5;

}
