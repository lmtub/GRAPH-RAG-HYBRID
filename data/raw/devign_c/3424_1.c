static void encode_quant_matrix(VC2EncContext *s)

{

    int level, custom_quant_matrix = 0;

    if (s->wavelet_depth > 4 || s->quant_matrix != VC2_QM_DEF)

        custom_quant_matrix = 1;

    put_bits(&s->pb, 1, custom_quant_matrix);

    if (custom_quant_matrix) {

        init_custom_qm(s);

        put_vc2_ue_uint(&s->pb, s->quant[0][0]);

        for (level = 0; level < s->wavelet_depth; level++) {

            put_vc2_ue_uint(&s->pb, s->quant[level][1]);

            put_vc2_ue_uint(&s->pb, s->quant[level][2]);

            put_vc2_ue_uint(&s->pb, s->quant[level][3]);

        }

    } else {

        for (level = 0; level < s->wavelet_depth; level++) {

            s->quant[level][0] = ff_dirac_default_qmat[s->wavelet_idx][level][0];

            s->quant[level][1] = ff_dirac_default_qmat[s->wavelet_idx][level][1];

            s->quant[level][2] = ff_dirac_default_qmat[s->wavelet_idx][level][2];

            s->quant[level][3] = ff_dirac_default_qmat[s->wavelet_idx][level][3];

        }

    }

}
