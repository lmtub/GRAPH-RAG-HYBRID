static int unpack_block_qpis(Vp3DecodeContext *s, GetBitContext *gb)

{

    int qpi, i, j, bit, run_length, blocks_decoded, num_blocks_at_qpi;

    int num_blocks = s->coded_fragment_list_index;



    for (qpi = 0; qpi < s->nqps-1 && num_blocks > 0; qpi++) {

        i = blocks_decoded = num_blocks_at_qpi = 0;



        bit = get_bits1(gb);



        do {

            run_length = get_vlc2(gb, s->superblock_run_length_vlc.table, 6, 2) + 1;

            if (run_length == 34)

                run_length += get_bits(gb, 12);

            blocks_decoded += run_length;



            if (!bit)

                num_blocks_at_qpi += run_length;



            for (j = 0; j < run_length; i++) {

                if (i > s->coded_fragment_list_index)

                    return -1;



                if (s->all_fragments[s->coded_fragment_list[i]].qpi == qpi) {

                    s->all_fragments[s->coded_fragment_list[i]].qpi += bit;

                    j++;

                }

            }



            if (run_length == 4129)

                bit = get_bits1(gb);

            else

                bit ^= 1;

        } while (blocks_decoded < num_blocks);



        num_blocks -= num_blocks_at_qpi;

    }



    return 0;

}
