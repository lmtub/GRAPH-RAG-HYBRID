static int _do_bit_allocation(AC3DecodeContext *ctx, int chnl)

{

    ac3_audio_block *ab = &ctx->audio_block;

    int16_t sdecay, fdecay, sgain, dbknee, floor;

    int16_t lowcomp, fgain, snroffset, fastleak, slowleak;

    int16_t psd[256], bndpsd[50], excite[50], mask[50], delta;

    uint8_t start, end, bin, i, j, k, lastbin, bndstrt, bndend, begin, deltnseg, band, seg, address;

    uint8_t fscod = ctx->sync_info.fscod;

    uint8_t *exps, *deltoffst, *deltlen, *deltba;

    uint8_t *baps;

    int do_delta = 0;



    /* initialization */

    sdecay = sdecaytab[ab->sdcycod];

    fdecay = fdecaytab[ab->fdcycod];

    sgain = sgaintab[ab->sgaincod];

    dbknee = dbkneetab[ab->dbpbcod];

    floor = floortab[ab->floorcod];



    if (chnl == 5) {

        start = ab->cplstrtmant;

        end = ab->cplendmant;

        fgain = fgaintab[ab->cplfgaincod];

        snroffset = (((ab->csnroffst - 15) << 4) + ab->cplfsnroffst) << 2;

        fastleak = (ab->cplfleak << 8) + 768;

        slowleak = (ab->cplsleak << 8) + 768;

        exps = ab->dcplexps;

        baps = ab->cplbap;

        if (ab->cpldeltbae == 0 || ab->cpldeltbae == 1) {

            do_delta = 1;

            deltnseg = ab->cpldeltnseg;

            deltoffst = ab->cpldeltoffst;

            deltlen = ab->cpldeltlen;

            deltba = ab->cpldeltba;

        }

    }

    else if (chnl == 6) {

        start = 0;

        end = 7;

        lowcomp = 0;

        fgain = fgaintab[ab->lfefgaincod];

        snroffset = (((ab->csnroffst - 15) << 4) + ab->lfefsnroffst) << 2;

        exps = ab->dlfeexps;

        baps = ab->lfebap;

    }

    else {

        start = 0;

        end = ab->endmant[chnl];

        lowcomp = 0;

        fgain = fgaintab[ab->fgaincod[chnl]];

        snroffset = (((ab->csnroffst - 15) << 4) + ab->fsnroffst[chnl]) << 2;

        exps = ab->dexps[chnl];

        baps = ab->bap[chnl];

        if (ab->deltbae[chnl] == 0 || ab->deltbae[chnl] == 1) {

            do_delta = 1;

            deltnseg = ab->deltnseg[chnl];

            deltoffst = ab->deltoffst[chnl];

            deltlen = ab->deltlen[chnl];

            deltba = ab->deltba[chnl];

        }

    }



    for (bin = start; bin < end; bin++) /* exponent mapping into psd */

        psd[bin] = (3072 - ((int16_t) (exps[bin] << 7)));



    /* psd integration */

    j = start;

    k = masktab[start];

    do {

        lastbin = FFMIN(bndtab[k] + bndsz[k], end);

        bndpsd[k] = psd[j];

        j++;

        for (i = j; i < lastbin; i++) {

            bndpsd[k] = logadd(bndpsd[k], psd[j]);

            j++;

        }

        k++;

    } while (end > lastbin);



    /* compute the excite function */

    bndstrt = masktab[start];

    bndend = masktab[end - 1] + 1;

    if (bndstrt == 0) {

        lowcomp = calc_lowcomp(lowcomp, bndpsd[0], bndpsd[1], 0);

        excite[0] = bndpsd[0] - fgain - lowcomp;

        lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);

        excite[1] = bndpsd[1] - fgain - lowcomp;

        begin = 7;

        for (bin = 2; bin < 7; bin++) {

            if (bndend != 7 || bin != 6)

                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);

            fastleak = bndpsd[bin] - fgain;

            slowleak = bndpsd[bin] - sgain;

            excite[bin] = fastleak - lowcomp;

            if (bndend != 7 || bin != 6)

                if (bndpsd[bin] <= bndpsd[bin + 1]) {

                    begin = bin + 1;

                    break;

                }

        }

        for (bin = begin; bin < (FFMIN(bndend, 22)); bin++) {

            if (bndend != 7 || bin != 6)

                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);

            fastleak -= fdecay;

            fastleak = FFMAX(fastleak, bndpsd[bin] - fgain);

            slowleak -= sdecay;

            slowleak = FFMAX(slowleak, bndpsd[bin] - sgain);

            excite[bin] = FFMAX(fastleak - lowcomp, slowleak);

        }

        begin = 22;

    }

    else {

        begin = bndstrt;

    }

    for (bin = begin; bin < bndend; bin++) {

        fastleak -= fdecay;

        fastleak = FFMAX(fastleak, bndpsd[bin] - fgain);

        slowleak -= sdecay;

        slowleak = FFMAX(slowleak, bndpsd[bin] - sgain);

        excite[bin] = FFMAX(fastleak, slowleak);

    }



    /* compute the masking curve */

    for (bin = bndstrt; bin < bndend; bin++) {

        if (bndpsd[bin] < dbknee)

            excite[bin] += ((dbknee - bndpsd[bin]) >> 2);

        mask[bin] = FFMAX(excite[bin], hth[bin][fscod]);

    }



    /* apply the delta bit allocation */

    if (do_delta) {

        band = 0;

        for (seg = 0; seg < deltnseg + 1; seg++) {

            band += deltoffst[seg];

            if (deltba[seg] >= 4)

                delta = (deltba[seg] - 3) << 7;

            else

                delta = (deltba[seg] - 4) << 7;

            for (k = 0; k < deltlen[seg]; k++) {

                mask[band] += delta;

                band++;

            }

        }

    }



    /*compute the bit allocation */

    i = start;

    j = masktab[start];

    do {

        lastbin = FFMIN(bndtab[j] + bndsz[j], end);

        mask[j] -= snroffset;

        mask[j] -= floor;

        if (mask[j] < 0)

            mask[j] = 0;

        mask[j] &= 0x1fe0;

        mask[j] += floor;

        for (k = i; k < lastbin; k++) {

            address = (psd[i] - mask[j]) >> 5;

            address = FFMIN(63, (FFMAX(0, address)));

            baps[i] = baptab[address];

            i++;

        }

        j++;

    } while (end > lastbin);



    return 0;

}
