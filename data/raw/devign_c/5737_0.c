static int jpeg2000_decode_packets(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)

{

    int ret, i;

    int tp_index = 0;



    s->bit_index = 8;

    if (tile->poc.nb_poc) {

        for (i=0; i<tile->poc.nb_poc; i++) {

            Jpeg2000POCEntry *e = &tile->poc.poc[i];

            ret = jpeg2000_decode_packets_po_iteration(s, tile,

                e->RSpoc, e->CSpoc,

                e->LYEpoc, e->REpoc, e->CEpoc,

                e->Ppoc, &tp_index

                );

            if (ret < 0)

                return ret;

        }

    } else {

        ret = jpeg2000_decode_packets_po_iteration(s, tile,

            0, 0,

            tile->codsty[0].nlayers,

            33,

            s->ncomponents,

            tile->codsty[0].prog_order,

            &tp_index

        );

    }

    /* EOC marker reached */

    bytestream2_skip(&s->g, 2);



    return ret;

}
