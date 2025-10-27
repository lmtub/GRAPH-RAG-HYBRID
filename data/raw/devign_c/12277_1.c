static int fdctrl_init_common(FDCtrl *fdctrl)

{

    int i, j;

    static int command_tables_inited = 0;



    /* Fill 'command_to_handler' lookup table */

    if (!command_tables_inited) {

        command_tables_inited = 1;

        for (i = ARRAY_SIZE(handlers) - 1; i >= 0; i--) {

            for (j = 0; j < sizeof(command_to_handler); j++) {

                if ((j & handlers[i].mask) == handlers[i].value) {

                    command_to_handler[j] = i;

                }

            }

        }

    }



    FLOPPY_DPRINTF("init controller\n");

    fdctrl->fifo = qemu_memalign(512, FD_SECTOR_LEN);

    fdctrl->fifo_size = 512;

    fdctrl->result_timer = qemu_new_timer(vm_clock,

                                          fdctrl_result_timer, fdctrl);



    fdctrl->version = 0x90; /* Intel 82078 controller */

    fdctrl->config = FD_CONFIG_EIS | FD_CONFIG_EFIFO; /* Implicit seek, polling & FIFO enabled */

    fdctrl->num_floppies = MAX_FD;



    if (fdctrl->dma_chann != -1)

        DMA_register_channel(fdctrl->dma_chann, &fdctrl_transfer_handler, fdctrl);

    fdctrl_connect_drives(fdctrl);



    return 0;

}
