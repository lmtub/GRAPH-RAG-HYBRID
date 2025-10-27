static int get_free_io_mem_idx(void)

{

    int i;



    for (i = 0; i<IO_MEM_NB_ENTRIES; i++)

        if (!io_mem_used[i]) {

            io_mem_used[i] = 1;

            return i;

        }



    return -1;

}
