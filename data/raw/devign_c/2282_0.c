static int advanced_decode_i_mbs(VC9Context *v)

{

  int i, x, y, cbpcy, mqdiff, absmq, mquant, ac_pred, condover,

    current_mb = 0, over_flags_mb = 0;



    for (y=0; y<v->height_mb; y++)

    {

        for (x=0; x<v->width_mb; x++)

        {

            if (v->ac_pred_plane[i])

                ac_pred = get_bits(&v->gb, 1);

            if (condover == 3 && v->over_flags_plane)

                over_flags_mb = get_bits(&v->gb, 1);

            GET_MQUANT();

        }

        current_mb++;

    }

    return 0;

}
