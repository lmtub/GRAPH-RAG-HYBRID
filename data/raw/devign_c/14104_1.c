static int kvm_check_many_ioeventfds(void)

{

    /* Userspace can use ioeventfd for io notification.  This requires a host

     * that supports eventfd(2) and an I/O thread; since eventfd does not

     * support SIGIO it cannot interrupt the vcpu.

     *

     * Older kernels have a 6 device limit on the KVM io bus.  Find out so we

     * can avoid creating too many ioeventfds.

     */

#if defined(CONFIG_EVENTFD) && defined(CONFIG_IOTHREAD)

    int ioeventfds[7];

    int i, ret = 0;

    for (i = 0; i < ARRAY_SIZE(ioeventfds); i++) {

        ioeventfds[i] = eventfd(0, EFD_CLOEXEC);

        if (ioeventfds[i] < 0) {

            break;

        }

        ret = kvm_set_ioeventfd_pio_word(ioeventfds[i], 0, i, true);

        if (ret < 0) {

            close(ioeventfds[i]);

            break;

        }

    }



    /* Decide whether many devices are supported or not */

    ret = i == ARRAY_SIZE(ioeventfds);



    while (i-- > 0) {

        kvm_set_ioeventfd_pio_word(ioeventfds[i], 0, i, false);

        close(ioeventfds[i]);

    }

    return ret;

#else

    return 0;

#endif

}
