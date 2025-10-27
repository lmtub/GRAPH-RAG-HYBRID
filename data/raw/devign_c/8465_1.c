static void init_custom_qm(VC2EncContext *s)

{

    int level, orientation;



    if (s->quant_matrix == VC2_QM_DEF) {

        for (level = 0; level < s->wavelet_depth; level++) {

            for (orientation = 0; orientation < 4; orientation++) {

                if (level <= 3)

                    s->quant[level][orientation] = ff_dirac_default_qmat[s->wavelet_idx][level][orientation];

                else

                    s->quant[level][orientation] = vc2_qm_col_tab[level][orientation];

            }

        }

    } else if (s->quant_matrix == VC2_QM_COL) {

        for (level = 0; level < s->wavelet_depth; level++) {

            for (orientation = 0; orientation < 4; orientation++) {

                s->quant[level][orientation] = vc2_qm_col_tab[level][orientation];

            }

        }

    } else {

        for (level = 0; level < s->wavelet_depth; level++) {

            for (orientation = 0; orientation < 4; orientation++) {

                s->quant[level][orientation] = vc2_qm_flat_tab[level][orientation];

            }

        }

    }

}
