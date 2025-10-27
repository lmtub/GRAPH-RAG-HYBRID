fdctrl_t *sun4m_fdctrl_init (qemu_irq irq, target_phys_addr_t io_base,

                             BlockDriverState **fds)

{

    fdctrl_t *fdctrl;



    fdctrl = fdctrl_init(irq, 0, 1, io_base, fds);

    fdctrl->sun4m = 1;



    return fdctrl;

}
