static void stereo_processing(PSContext *ps, float (*l)[32][2], float (*r)[32][2], int is34)

{

    int e, b, k, n;



    float (*H11)[PS_MAX_NUM_ENV+1][PS_MAX_NR_IIDICC] = ps->H11;

    float (*H12)[PS_MAX_NUM_ENV+1][PS_MAX_NR_IIDICC] = ps->H12;

    float (*H21)[PS_MAX_NUM_ENV+1][PS_MAX_NR_IIDICC] = ps->H21;

    float (*H22)[PS_MAX_NUM_ENV+1][PS_MAX_NR_IIDICC] = ps->H22;

    int8_t *opd_hist = ps->opd_hist;

    int8_t *ipd_hist = ps->ipd_hist;

    int8_t iid_mapped_buf[PS_MAX_NUM_ENV][PS_MAX_NR_IIDICC];

    int8_t icc_mapped_buf[PS_MAX_NUM_ENV][PS_MAX_NR_IIDICC];

    int8_t ipd_mapped_buf[PS_MAX_NUM_ENV][PS_MAX_NR_IIDICC];

    int8_t opd_mapped_buf[PS_MAX_NUM_ENV][PS_MAX_NR_IIDICC];

    int8_t (*iid_mapped)[PS_MAX_NR_IIDICC] = iid_mapped_buf;

    int8_t (*icc_mapped)[PS_MAX_NR_IIDICC] = icc_mapped_buf;

    int8_t (*ipd_mapped)[PS_MAX_NR_IIDICC] = ipd_mapped_buf;

    int8_t (*opd_mapped)[PS_MAX_NR_IIDICC] = opd_mapped_buf;

    const int8_t *k_to_i = is34 ? k_to_i_34 : k_to_i_20;

    const float (*H_LUT)[8][4] = (PS_BASELINE || ps->icc_mode < 3) ? HA : HB;



    //Remapping

    if (ps->num_env_old) {

        memcpy(H11[0][0], H11[0][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H11[0][0][0]));

        memcpy(H11[1][0], H11[1][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H11[1][0][0]));

        memcpy(H12[0][0], H12[0][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H12[0][0][0]));

        memcpy(H12[1][0], H12[1][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H12[1][0][0]));

        memcpy(H21[0][0], H21[0][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H21[0][0][0]));

        memcpy(H21[1][0], H21[1][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H21[1][0][0]));

        memcpy(H22[0][0], H22[0][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H22[0][0][0]));

        memcpy(H22[1][0], H22[1][ps->num_env_old], PS_MAX_NR_IIDICC*sizeof(H22[1][0][0]));

    }



    if (is34) {

        remap34(&iid_mapped, ps->iid_par, ps->nr_iid_par, ps->num_env, 1);

        remap34(&icc_mapped, ps->icc_par, ps->nr_icc_par, ps->num_env, 1);

        if (ps->enable_ipdopd) {

            remap34(&ipd_mapped, ps->ipd_par, ps->nr_ipdopd_par, ps->num_env, 0);

            remap34(&opd_mapped, ps->opd_par, ps->nr_ipdopd_par, ps->num_env, 0);

        }

        if (!ps->is34bands_old) {

            map_val_20_to_34(H11[0][0]);

            map_val_20_to_34(H11[1][0]);

            map_val_20_to_34(H12[0][0]);

            map_val_20_to_34(H12[1][0]);

            map_val_20_to_34(H21[0][0]);

            map_val_20_to_34(H21[1][0]);

            map_val_20_to_34(H22[0][0]);

            map_val_20_to_34(H22[1][0]);

            ipdopd_reset(ipd_hist, opd_hist);

        }

    } else {

        remap20(&iid_mapped, ps->iid_par, ps->nr_iid_par, ps->num_env, 1);

        remap20(&icc_mapped, ps->icc_par, ps->nr_icc_par, ps->num_env, 1);

        if (ps->enable_ipdopd) {

            remap20(&ipd_mapped, ps->ipd_par, ps->nr_ipdopd_par, ps->num_env, 0);

            remap20(&opd_mapped, ps->opd_par, ps->nr_ipdopd_par, ps->num_env, 0);

        }

        if (ps->is34bands_old) {

            map_val_34_to_20(H11[0][0]);

            map_val_34_to_20(H11[1][0]);

            map_val_34_to_20(H12[0][0]);

            map_val_34_to_20(H12[1][0]);

            map_val_34_to_20(H21[0][0]);

            map_val_34_to_20(H21[1][0]);

            map_val_34_to_20(H22[0][0]);

            map_val_34_to_20(H22[1][0]);

            ipdopd_reset(ipd_hist, opd_hist);

        }

    }



    //Mixing

    for (e = 0; e < ps->num_env; e++) {

        for (b = 0; b < NR_PAR_BANDS[is34]; b++) {

            float h11, h12, h21, h22;

            h11 = H_LUT[iid_mapped[e][b] + 7 + 23 * ps->iid_quant][icc_mapped[e][b]][0];

            h12 = H_LUT[iid_mapped[e][b] + 7 + 23 * ps->iid_quant][icc_mapped[e][b]][1];

            h21 = H_LUT[iid_mapped[e][b] + 7 + 23 * ps->iid_quant][icc_mapped[e][b]][2];

            h22 = H_LUT[iid_mapped[e][b] + 7 + 23 * ps->iid_quant][icc_mapped[e][b]][3];

            if (!PS_BASELINE && ps->enable_ipdopd && b < ps->nr_ipdopd_par) {

                //The spec say says to only run this smoother when enable_ipdopd

                //is set but the reference decoder appears to run it constantly

                float h11i, h12i, h21i, h22i;

                float ipd_adj_re, ipd_adj_im;

                int opd_idx = opd_hist[b] * 8 + opd_mapped[e][b];

                int ipd_idx = ipd_hist[b] * 8 + ipd_mapped[e][b];

                float opd_re = pd_re_smooth[opd_idx];

                float opd_im = pd_im_smooth[opd_idx];

                float ipd_re = pd_re_smooth[ipd_idx];

                float ipd_im = pd_im_smooth[ipd_idx];

                opd_hist[b] = opd_idx & 0x3F;

                ipd_hist[b] = ipd_idx & 0x3F;



                ipd_adj_re = opd_re*ipd_re + opd_im*ipd_im;

                ipd_adj_im = opd_im*ipd_re - opd_re*ipd_im;

                h11i = h11 * opd_im;

                h11  = h11 * opd_re;

                h12i = h12 * ipd_adj_im;

                h12  = h12 * ipd_adj_re;

                h21i = h21 * opd_im;

                h21  = h21 * opd_re;

                h22i = h22 * ipd_adj_im;

                h22  = h22 * ipd_adj_re;

                H11[1][e+1][b] = h11i;

                H12[1][e+1][b] = h12i;

                H21[1][e+1][b] = h21i;

                H22[1][e+1][b] = h22i;

            }

            H11[0][e+1][b] = h11;

            H12[0][e+1][b] = h12;

            H21[0][e+1][b] = h21;

            H22[0][e+1][b] = h22;

        }

        for (k = 0; k < NR_BANDS[is34]; k++) {

            float h11r, h12r, h21r, h22r;

            float h11i, h12i, h21i, h22i;

            float h11r_step, h12r_step, h21r_step, h22r_step;

            float h11i_step, h12i_step, h21i_step, h22i_step;

            int start = ps->border_position[e];

            int stop  = ps->border_position[e+1];

            float width = 1.f / (stop - start);

            b = k_to_i[k];

            h11r = H11[0][e][b];

            h12r = H12[0][e][b];

            h21r = H21[0][e][b];

            h22r = H22[0][e][b];

            if (!PS_BASELINE && ps->enable_ipdopd) {

            //Is this necessary? ps_04_new seems unchanged

            if ((is34 && k <= 13 && k >= 9) || (!is34 && k <= 1)) {

                h11i = -H11[1][e][b];

                h12i = -H12[1][e][b];

                h21i = -H21[1][e][b];

                h22i = -H22[1][e][b];

            } else {

                h11i = H11[1][e][b];

                h12i = H12[1][e][b];

                h21i = H21[1][e][b];

                h22i = H22[1][e][b];

            }

            }

            //Interpolation

            h11r_step = (H11[0][e+1][b] - h11r) * width;

            h12r_step = (H12[0][e+1][b] - h12r) * width;

            h21r_step = (H21[0][e+1][b] - h21r) * width;

            h22r_step = (H22[0][e+1][b] - h22r) * width;

            if (!PS_BASELINE && ps->enable_ipdopd) {

                h11i_step = (H11[1][e+1][b] - h11i) * width;

                h12i_step = (H12[1][e+1][b] - h12i) * width;

                h21i_step = (H21[1][e+1][b] - h21i) * width;

                h22i_step = (H22[1][e+1][b] - h22i) * width;

            }

            for (n = start + 1; n <= stop; n++) {

                //l is s, r is d

                float l_re = l[k][n][0];

                float l_im = l[k][n][1];

                float r_re = r[k][n][0];

                float r_im = r[k][n][1];

                h11r += h11r_step;

                h12r += h12r_step;

                h21r += h21r_step;

                h22r += h22r_step;

                if (!PS_BASELINE && ps->enable_ipdopd) {

                    h11i += h11i_step;

                    h12i += h12i_step;

                    h21i += h21i_step;

                    h22i += h22i_step;



                    l[k][n][0] = h11r*l_re + h21r*r_re - h11i*l_im - h21i*r_im;

                    l[k][n][1] = h11r*l_im + h21r*r_im + h11i*l_re + h21i*r_re;

                    r[k][n][0] = h12r*l_re + h22r*r_re - h12i*l_im - h22i*r_im;

                    r[k][n][1] = h12r*l_im + h22r*r_im + h12i*l_re + h22i*r_re;

                } else {

                    l[k][n][0] = h11r*l_re + h21r*r_re;

                    l[k][n][1] = h11r*l_im + h21r*r_im;

                    r[k][n][0] = h12r*l_re + h22r*r_re;

                    r[k][n][1] = h12r*l_im + h22r*r_im;

                }

            }

        }

    }

}
