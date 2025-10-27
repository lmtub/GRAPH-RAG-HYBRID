static int generate_coupling_coordinates(AC3DecodeContext * ctx)

{

    ac3_audio_block *ab = &ctx->audio_block;

    uint8_t exp, mstrcplco;

    int16_t mant;

    uint32_t cplbndstrc = (1 << ab->ncplsubnd) >> 1;

    int ch, bnd, sbnd;

    float cplco;



    if (ab->cplcoe)

        for (ch = 0; ch < ctx->bsi.nfchans; ch++)

            if (ab->cplcoe & (1 << ch)) {

                mstrcplco = 3 * ab->mstrcplco[ch];

                sbnd = ab->cplbegf;

                for (bnd = 0; bnd < ab->ncplbnd; bnd++) {

                    exp = ab->cplcoexp[ch][bnd];

                    if (exp == 15)

                        mant = ab->cplcomant[ch][bnd] <<= 14;

                    else

                        mant = (ab->cplcomant[ch][bnd] | 0x10) << 13;

                    cplco = to_float(exp + mstrcplco, mant);

                    if (ctx->bsi.acmod == 0x02 && (ab->flags & AC3_AB_PHSFLGINU) && ch == 1

                            && (ab->phsflg & (1 << bnd)))

                        cplco = -cplco; /* invert the right channel */

                    ab->cplco[ch][sbnd++] = cplco;

                    while (cplbndstrc & ab->cplbndstrc) {

                        cplbndstrc >>= 1;

                        ab->cplco[ch][sbnd++] = cplco;

                    }

                    cplbndstrc >>= 1;

                }

            }



    return 0;

}
