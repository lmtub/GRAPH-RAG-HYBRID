int ff_ivi_dec_huff_desc(GetBitContext *gb, int desc_coded, int which_tab,

                         IVIHuffTab *huff_tab, AVCodecContext *avctx)

{

    int         i, result;

    IVIHuffDesc new_huff;



    if (!desc_coded) {

        /* select default table */

        huff_tab->tab = (which_tab) ? &ff_ivi_blk_vlc_tabs[7]

            : &ff_ivi_mb_vlc_tabs [7];

    } else {

        huff_tab->tab_sel = get_bits(gb, 3);

        if (huff_tab->tab_sel == 7) {

            /* custom huffman table (explicitly encoded) */

            new_huff.num_rows = get_bits(gb, 4);



            for (i = 0; i < new_huff.num_rows; i++)

                new_huff.xbits[i] = get_bits(gb, 4);



            /* Have we got the same custom table? Rebuild if not. */

            if (ff_ivi_huff_desc_cmp(&new_huff, &huff_tab->cust_desc)) {

                ff_ivi_huff_desc_copy(&huff_tab->cust_desc, &new_huff);



                if (huff_tab->cust_tab.table)

                    ff_free_vlc(&huff_tab->cust_tab);

                result = ff_ivi_create_huff_from_desc(&huff_tab->cust_desc,

                        &huff_tab->cust_tab, 0);

                if (result) {

                    av_log(avctx, AV_LOG_ERROR,

                           "Error while initializing custom vlc table!\n");

                    return -1;

                }

            }

            huff_tab->tab = &huff_tab->cust_tab;

        } else {

            /* select one of predefined tables */

            huff_tab->tab = (which_tab) ? &ff_ivi_blk_vlc_tabs[huff_tab->tab_sel]

                : &ff_ivi_mb_vlc_tabs [huff_tab->tab_sel];

        }

    }



    return 0;

}
