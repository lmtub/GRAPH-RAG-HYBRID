static int sbr_hf_calc_npatches(AACContext *ac, SpectralBandReplication *sbr)

{

    int i, k, sb = 0;

    int msb = sbr->k[0];

    int usb = sbr->kx[1];

    int goal_sb = ((1000 << 11) + (sbr->sample_rate >> 1)) / sbr->sample_rate;



    sbr->num_patches = 0;



    if (goal_sb < sbr->kx[1] + sbr->m[1]) {

        for (k = 0; sbr->f_master[k] < goal_sb; k++) ;

    } else

        k = sbr->n_master;



    do {

        int odd = 0;

        for (i = k; i == k || sb > (sbr->k[0] - 1 + msb - odd); i--) {

            sb = sbr->f_master[i];

            odd = (sb + sbr->k[0]) & 1;

        }



        // Requirements (14496-3 sp04 p205) sets the maximum number of patches to 5.

        // After this check the final number of patches can still be six which is

        // illegal however the Coding Technologies decoder check stream has a final

        // count of 6 patches

        if (sbr->num_patches > 5) {

            av_log(ac->avctx, AV_LOG_ERROR, "Too many patches: %d\n", sbr->num_patches);

            return -1;

        }



        sbr->patch_num_subbands[sbr->num_patches]  = FFMAX(sb - usb, 0);

        sbr->patch_start_subband[sbr->num_patches] = sbr->k[0] - odd - sbr->patch_num_subbands[sbr->num_patches];



        if (sbr->patch_num_subbands[sbr->num_patches] > 0) {

            usb = sb;

            msb = sb;

            sbr->num_patches++;

        } else

            msb = sbr->kx[1];



        if (sbr->f_master[k] - sb < 3)

            k = sbr->n_master;

    } while (sb != sbr->kx[1] + sbr->m[1]);



    if (sbr->patch_num_subbands[sbr->num_patches-1] < 3 && sbr->num_patches > 1)

        sbr->num_patches--;



    return 0;

}
