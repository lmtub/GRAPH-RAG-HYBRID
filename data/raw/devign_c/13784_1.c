static void debug_print_fis(uint8_t *fis, int cmd_len)

{

#ifdef DEBUG_AHCI

    int i;



    fprintf(stderr, "fis:");

    for (i = 0; i < cmd_len; i++) {

        if ((i & 0xf) == 0) {

            fprintf(stderr, "\n%02x:",i);

        }

        fprintf(stderr, "%02x ",fis[i]);

    }

    fprintf(stderr, "\n");

#endif

}
