static int vfio_set_trigger_eventfd(VFIOINTp *intp,

                                    eventfd_user_side_handler_t handler)

{

    VFIODevice *vbasedev = &intp->vdev->vbasedev;

    struct vfio_irq_set *irq_set;

    int argsz, ret;

    int32_t *pfd;



    argsz = sizeof(*irq_set) + sizeof(*pfd);

    irq_set = g_malloc0(argsz);

    irq_set->argsz = argsz;

    irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;

    irq_set->index = intp->pin;

    irq_set->start = 0;

    irq_set->count = 1;

    pfd = (int32_t *)&irq_set->data;

    *pfd = event_notifier_get_fd(&intp->interrupt);

    qemu_set_fd_handler(*pfd, (IOHandler *)handler, NULL, intp);

    ret = ioctl(vbasedev->fd, VFIO_DEVICE_SET_IRQS, irq_set);

    g_free(irq_set);

    if (ret < 0) {

        error_report("vfio: Failed to set trigger eventfd: %m");

        qemu_set_fd_handler(*pfd, NULL, NULL, NULL);

    }

    return ret;

}
