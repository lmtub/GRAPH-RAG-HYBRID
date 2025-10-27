static void do_info_network(int argc, const char **argv)

{

    int i, j;

    NetDriverState *nd;

    

    for(i = 0; i < nb_nics; i++) {

        nd = &nd_table[i];

        term_printf("%d: ifname=%s macaddr=", i, nd->ifname);

        for(j = 0; j < 6; j++) {

            if (j > 0)

                term_printf(":");

            term_printf("%02x", nd->macaddr[j]);

        }

        term_printf("\n");

    }

}
