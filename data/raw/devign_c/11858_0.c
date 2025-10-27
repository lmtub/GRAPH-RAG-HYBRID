static int uncouple_channels(AC3DecodeContext * ctx)

{

    ac3_audio_block *ab = &ctx->audio_block;

    int ch, sbnd, bin;

    int index;

    float (*samples)[256];

    int16_t mantissa;



    samples = (float (*)[256])((ctx->bsi.flags & AC3_BSI_LFEON) ? (ctx->samples + 256) : (ctx->samples));



    /* uncouple channels */

    for (ch = 0; ch < ctx->bsi.nfchans; ch++)

        if (ab->chincpl & (1 << ch))

            for (sbnd = ab->cplbegf; sbnd < 3 + ab->cplendf; sbnd++)

                for (bin = 0; bin < 12; bin++) {

                    index = sbnd * 12 + bin + 37;

                    samples[ch][index] = ab->cplcoeffs[index] * ab->cplco[ch][sbnd] * ab->chcoeffs[ch];

                }



    /* generate dither if required */

    for (ch = 0; ch < ctx->bsi.nfchans; ch++)

        if ((ab->chincpl & (1 << ch)) && (ab->dithflag & (1 << ch)))

            for (index = 0; index < ab->endmant[ch]; index++)

                if (!ab->bap[ch][index]) {

                    mantissa = dither_int16(&ctx->state);

                    samples[ch][index] = to_float(ab->dexps[ch][index], mantissa) * ab->chcoeffs[ch];

                }



    return 0;

}
