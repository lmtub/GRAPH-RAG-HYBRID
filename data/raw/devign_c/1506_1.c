static int unpack_block_qpis(Vp3DecodeContext *s, GetBitContext *gb)

{

    int qpi, i, j, bit, run_length, blocks_decoded, num_blocks_at_qpi;

    int num_blocks = s->total_num_coded_frags;



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

                if (i >= s->total_num_coded_frags)

                    return -1;



                if (s->all_fragments[s->coded_fragment_list[0][i]].qpi == qpi) {

                    s->all_fragments[s->coded_fragment_list[0][i]].qpi += bit;

                    j++;

                }

            }



            if (run_length == MAXIMUM_LONG_BIT_RUN)

                bit = get_bits1(gb);

            else

                bit ^= 1;

        } while (blocks_decoded < num_blocks);



        num_blocks -= num_blocks_at_qpi;

    }



    return 0;

}
