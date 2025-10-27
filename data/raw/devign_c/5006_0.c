static void decode_parameters(SiprParameters* parms, GetBitContext *pgb,

                              const SiprModeParam *p)

{

    int i, j;



    parms->ma_pred_switch           = get_bits(pgb, p->ma_predictor_bits);



    for (i = 0; i < 5; i++)

        parms->vq_indexes[i]        = get_bits(pgb, p->vq_indexes_bits[i]);



    for (i = 0; i < p->subframe_count; i++) {

        parms->pitch_delay[i]       = get_bits(pgb, p->pitch_delay_bits[i]);

        parms->gp_index[i]          = get_bits(pgb, p->gp_index_bits);



        for (j = 0; j < p->number_of_fc_indexes; j++)

            parms->fc_indexes[i][j] = get_bits(pgb, p->fc_index_bits[j]);



        parms->gc_index[i]          = get_bits(pgb, p->gc_index_bits);

    }

}
