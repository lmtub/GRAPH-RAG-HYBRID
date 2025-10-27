static void fdctrl_raise_irq(FDCtrl *fdctrl, uint8_t status0)

{

    /* Sparc mutation */

    if (fdctrl->sun4m && (fdctrl->msr & FD_MSR_CMDBUSY)) {

        /* XXX: not sure */

        fdctrl->msr &= ~FD_MSR_CMDBUSY;

        fdctrl->msr |= FD_MSR_RQM | FD_MSR_DIO;

        fdctrl->status0 = status0;

        return;

    }

    if (!(fdctrl->sra & FD_SRA_INTPEND)) {

        qemu_set_irq(fdctrl->irq, 1);

        fdctrl->sra |= FD_SRA_INTPEND;

    }

    if (status0 & FD_SR0_SEEK) {

        FDrive *cur_drv;

        /* A seek clears the disk change line (if a disk is inserted) */

        cur_drv = get_cur_drv(fdctrl);

        if (cur_drv->max_track) {

            cur_drv->media_changed = 0;

        }

    }



    fdctrl->reset_sensei = 0;

    fdctrl->status0 = status0;

    FLOPPY_DPRINTF("Set interrupt status to 0x%02x\n", fdctrl->status0);

}
