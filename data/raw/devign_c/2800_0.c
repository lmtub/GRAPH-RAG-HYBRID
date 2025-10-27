static void page_flush_tb(void)

{

    int i;



    for (i = 0; i < V_L1_SIZE; i++) {

        page_flush_tb_1(V_L1_SHIFT / L2_BITS - 1, l1_map + i);

    }

}
