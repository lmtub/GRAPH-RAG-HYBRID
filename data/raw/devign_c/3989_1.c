bool qvirtio_wait_queue_isr(const QVirtioBus *bus, QVirtioDevice *d,

                                            QVirtQueue *vq, uint64_t timeout)

{

    do {

        clock_step(100);

        if (bus->get_queue_isr_status(d, vq)) {

            break; /* It has ended */

        }

    } while (--timeout);



    return timeout != 0;

}
