void net_set_boot_mask(int net_boot_mask)

{

    int i;



    /* Only the first four NICs may be bootable */

    net_boot_mask = net_boot_mask & 0xF;



    for (i = 0; i < nb_nics; i++) {

        if (net_boot_mask & (1 << i)) {

            net_boot_mask &= ~(1 << i);

        }

    }



    if (net_boot_mask) {

        fprintf(stderr, "Cannot boot from non-existent NIC\n");

        exit(1);

    }

}
