static void do_commit(int argc, const char **argv)

{

    int i;



    for (i = 0; i < MAX_DISKS; i++) {

        if (bs_table[i])

            bdrv_commit(bs_table[i]);

    }

}
