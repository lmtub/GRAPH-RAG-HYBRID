static uint32_t fdctrl_read_data (fdctrl_t *fdctrl)

{

    fdrive_t *cur_drv;

    uint32_t retval = 0;

    int pos, len;



    cur_drv = get_cur_drv(fdctrl);

    fdctrl->state &= ~FD_CTRL_SLEEP;

    if (FD_STATE(fdctrl->data_state) == FD_STATE_CMD) {

        FLOPPY_ERROR("can't read data in CMD state\n");

        return 0;

    }

    pos = fdctrl->data_pos;

    if (FD_STATE(fdctrl->data_state) == FD_STATE_DATA) {

        pos %= FD_SECTOR_LEN;

        if (pos == 0) {

            len = fdctrl->data_len - fdctrl->data_pos;

            if (len > FD_SECTOR_LEN)

                len = FD_SECTOR_LEN;

            bdrv_read(cur_drv->bs, fd_sector(cur_drv),

                      fdctrl->fifo, len);

        }

    }

    retval = fdctrl->fifo[pos];

    if (++fdctrl->data_pos == fdctrl->data_len) {

        fdctrl->data_pos = 0;

        /* Switch from transfer mode to status mode

         * then from status mode to command mode

         */

        if (FD_STATE(fdctrl->data_state) == FD_STATE_DATA) {

            fdctrl_stop_transfer(fdctrl, 0x20, 0x00, 0x00);

        } else {

            fdctrl_reset_fifo(fdctrl);

            fdctrl_reset_irq(fdctrl);

        }

    }

    FLOPPY_DPRINTF("data register: 0x%02x\n", retval);



    return retval;

}
